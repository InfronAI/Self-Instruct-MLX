"""Bounded, resumable Self-Instruct: seeds → tasks → labels → instances → training."""

import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import (
    environment,
    event,
    file_sha256,
    read_jsonl,
    run_lock,
    utc_now,
    write_json,
    write_jsonl,
)
from .backends import DEFAULT_MODEL, DEFAULT_REVISION, MLXBackend, RequestStore, parse_json
from .filtering import NoveltyIndex, clean_instances, instruction_problem
from .prompts import (
    chat_record,
    classification_messages,
    instance_messages,
    instruction_messages,
    user_prompt,
)


@dataclass
class RunConfig:
    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    seed_tasks: str = "data/seed_tasks.jsonl"
    seed: int = 42
    num_instructions: int = 12
    instructions_per_request: int = 4
    prompt_examples: int = 6
    instances_per_instruction: int = 2
    max_rounds: int = 12
    max_attempts: int = 2
    rouge_threshold: float = 0.7
    generation_max_tokens: int = 768
    eval_tasks: int = 8
    validation_tasks: int = 16
    include_seed_tasks: bool = True
    train_iters: int = 20
    learning_rate: float = 1e-5
    lora_layers: int = 4
    lora_rank: int = 8
    max_seq_length: int = 512
    eval_max_tokens: int = 128

    def validate(self):
        for key in (
            "num_instructions",
            "instructions_per_request",
            "prompt_examples",
            "instances_per_instruction",
            "max_rounds",
            "max_attempts",
            "generation_max_tokens",
            "eval_tasks",
            "validation_tasks",
            "train_iters",
            "lora_layers",
            "lora_rank",
            "max_seq_length",
            "eval_max_tokens",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.rouge_threshold < 1 or self.learning_rate <= 0:
            raise ValueError("Require 0 < rouge_threshold < 1 and learning_rate > 0")
        if self.instances_per_instruction > 5:
            raise ValueError("Self-Instruct retains at most 5 instances per instruction")
        if self.prompt_examples < 3 or self.max_seq_length < 64:
            raise ValueError("Require prompt_examples >= 3 and max_seq_length >= 64")


def split_seeds(seeds, config):
    if len(seeds) < config.eval_tasks + config.validation_tasks + config.prompt_examples:
        raise ValueError("Not enough seed tasks for disjoint train/validation/test pools")
    instructions = [s["instruction"].strip() for s in seeds]
    if len(set(instructions)) != len(instructions):
        raise ValueError("Seed task instructions must be unique before splitting")
    shuffled = list(seeds)
    random.Random(config.seed).shuffle(shuffled)
    test = shuffled[: config.eval_tasks]
    valid = shuffled[config.eval_tasks : config.eval_tasks + config.validation_tasks]
    train = shuffled[config.eval_tasks + config.validation_tasks :]
    return train, valid, test


def request_json(store, directory, stage, messages, config, validate, seed, max_tokens=None):
    for attempt in range(config.max_attempts):
        record = store.request(
            stage,
            messages,
            max_tokens=max_tokens or config.generation_max_tokens,
            temperature=0.7 if stage == "instructions" else 0.0,
            seed=seed + attempt,
        )
        try:
            if record["response"]["finish_reason"] == "length":
                raise ValueError("truncated_response")
            value = parse_json(record["response"]["text"])
            # Small local models may use a bare array or wrap instruction strings
            # in objects. Normalize only explicit, unambiguous fields; never invent values.
            if stage == "instructions":
                items = value.get("instructions") if isinstance(value, dict) else value
                if isinstance(items, list):
                    items = [
                        item.get("instruction", item.get("instructions"))
                        if isinstance(item, dict)
                        else item
                        for item in items
                    ]
                    value = {"instructions": items}
            elif stage == "instances" and isinstance(value, list):
                value = {"instances": value}
            if not validate(value):
                raise ValueError("invalid_schema")
            return value, record["key"]
        except (ValueError, TypeError) as error:
            event(directory, stage, f"Response rejected: {error}", request_key=record["key"])
            messages = messages + [
                {"role": "assistant", "content": record["response"]["text"]},
                {
                    "role": "user",
                    "content": "Return only complete JSON with exactly the schema in the original request. Keep values brief; no extra fields.",
                },
            ]
    return None, record["key"]


def generate_data(directory, config, store):
    seeds = read_jsonl(config.seed_tasks)
    train_seeds, valid_seeds, test_seeds = split_seeds(seeds, config)
    write_json(
        directory / "seed_split.json",
        {
            name: [s["id"] for s in subset]
            for name, subset in [
                ("train", train_seeds),
                ("valid", valid_seeds),
                ("test", test_seeds),
            ]
        },
    )
    # Only train seeds enter prompts. All seeds enter novelty filtering to protect held-out tasks.
    index = NoveltyIndex(s["instruction"] for s in seeds)
    instructions, rejected = [], []
    rng = random.Random(config.seed)
    event(
        directory,
        "instructions",
        f"Bootstrapping {config.num_instructions} tasks from {len(train_seeds)} training seeds",
    )
    for round_id in range(config.max_rounds):
        machine_examples = rng.sample(instructions, min(2, len(instructions)))
        human_examples = rng.sample(train_seeds, config.prompt_examples - len(machine_examples))
        examples = [t["instruction"] for t in machine_examples + human_examples]
        rng.shuffle(examples)
        value, key = request_json(
            store,
            directory,
            "instructions",
            instruction_messages(examples, config.instructions_per_request),
            config,
            lambda v: (
                isinstance(v, dict)
                and isinstance(v.get("instructions"), list)
                and all(isinstance(i, str) for i in v["instructions"])
            ),
            config.seed + round_id * 100,
        )
        if value:
            for candidate in value["instructions"]:
                instruction = " ".join(candidate.split()).strip()
                problem = instruction_problem(instruction)
                similarity = index.compare(instruction) if not problem else {}
                if not problem and similarity["max_similarity"] > config.rouge_threshold:
                    problem = "rouge_overlap"
                if problem:
                    rejected.append(
                        {
                            "instruction": instruction,
                            "reason": problem,
                            "round": round_id,
                            "request_key": key,
                            **similarity,
                        }
                    )
                    continue
                if len(instructions) >= config.num_instructions:
                    break
                instructions.append(
                    {
                        "id": f"generated_{len(instructions):04d}",
                        "instruction": instruction,
                        "round": round_id,
                        "request_key": key,
                        **similarity,
                    }
                )
                index.add(instruction)
        write_jsonl(directory / "instructions.jsonl", instructions)
        write_jsonl(directory / "rejected_instructions.jsonl", rejected)
        event(
            directory,
            "instructions",
            f"Round {round_id + 1}: {len(instructions)}/{config.num_instructions} accepted; {len(rejected)} filtered",
        )
        if len(instructions) == config.num_instructions:
            break
    if len(instructions) < config.num_instructions:
        raise RuntimeError(
            f"Generation budget exhausted: {len(instructions)}/{config.num_instructions} instructions. "
            "Inspect requests and rejections; use a new run/config for a larger budget."
        )

    classifications, generated, rejected_instances = [], [], []
    for i, task in enumerate(instructions):
        instruction = task["instruction"]
        value, key = request_json(
            store,
            directory,
            "classification",
            classification_messages(instruction),
            config,
            lambda v: isinstance(v, dict) and type(v.get("is_classification")) is bool,
            config.seed + 10000 + i * 100,
            max_tokens=64,
        )
        if value is None:
            rejected_instances.append(
                {"task_id": task["id"], "reason": "classification_failed", "request_key": key}
            )
            continue
        is_classification = value["is_classification"]
        classifications.append(
            {**task, "is_classification": is_classification, "classification_request": key}
        )
        value, key = request_json(
            store,
            directory,
            "instances",
            instance_messages(instruction, is_classification, config.instances_per_instruction),
            config,
            lambda v: isinstance(v, dict) and isinstance(v.get("instances"), list),
            config.seed + 20000 + i * 100,
        )
        valid, invalid = clean_instances(
            instruction, value["instances"] if value else [], config.instances_per_instruction
        )
        if not valid:
            rejected_instances.append(
                {"task_id": task["id"], "reason": "no_valid_instances", "request_key": key}
            )
        for row in valid:
            generated.append(
                {
                    **row,
                    "task_id": task["id"],
                    "source": "generated",
                    "is_classification": is_classification,
                    "request_key": key,
                }
            )
        rejected_instances.extend({"task_id": task["id"], **v} for v in invalid)
        write_jsonl(directory / "classifications.jsonl", classifications)
        write_jsonl(directory / "instances.jsonl", generated)
        write_jsonl(directory / "rejected_instances.jsonl", rejected_instances)
        event(
            directory,
            "instances",
            f"Task {i + 1}/{len(instructions)}: {'output-first' if is_classification else 'input-first'}; {len(valid)} examples retained",
        )
    if len({row["task_id"] for row in generated}) < 2:
        raise RuntimeError(
            "Fewer than two generated tasks have valid instances; inspect raw responses"
        )

    # Split at task level, never at example level, to keep sibling instances together.
    generated_ids = sorted({r["task_id"] for r in generated})
    rng = random.Random(config.seed)
    rng.shuffle(generated_ids)
    validation_ids = set(generated_ids[: max(1, len(generated_ids) // 5)])
    splits = {"train": [], "valid": [], "test": []}
    for row in generated:
        splits["valid" if row["task_id"] in validation_ids else "train"].append(row)
    seed_rejections = []
    for name, subset in [("train", train_seeds), ("valid", valid_seeds), ("test", test_seeds)]:
        if name == "train" and not config.include_seed_tasks:
            continue
        for task in subset:
            # Preserve human references verbatim. The original generated-output
            # colon heuristic would incorrectly drop valid cloze/analogy tests.
            valid = []
            for row in task["instances"]:
                if (
                    isinstance(row.get("input"), str)
                    and isinstance(row.get("output"), str)
                    and row["output"].strip()
                ):
                    valid.append({"instruction": task["instruction"], **row})
                else:
                    seed_rejections.append(
                        {"task_id": task["id"], "split": name, "reason": "invalid_seed_schema"}
                    )
            for row in valid:
                splits[name].append(
                    {
                        **row,
                        "task_id": task["id"],
                        "source": "seed",
                        "is_classification": task["is_classification"],
                    }
                )
    write_jsonl(directory / "rejected_seeds.jsonl", seed_rejections)
    if any(not rows for rows in splits.values()):
        raise RuntimeError("An empty dataset split remains after filtering")
    instruction_sets = [{r["instruction"] for r in rows} for rows in splits.values()]
    if any(a & b for i, a in enumerate(instruction_sets) for b in instruction_sets[i + 1 :]):
        raise RuntimeError("Task leakage between dataset splits")
    for name, rows in splits.items():
        write_jsonl(directory / "dataset" / f"{name}_provenance.jsonl", rows)
        write_jsonl(directory / "dataset" / f"{name}.jsonl", [chat_record(row) for row in rows])
    write_jsonl(
        directory / "dataset" / "prompt_completion.jsonl",
        [
            {
                "prompt": user_prompt(row["instruction"], row["input"]) + "\n\nOutput:",
                "completion": " " + row["output"] + "<|endoftext|>",
            }
            for row in splits["train"]
        ],
    )
    summary = {
        "seed_tasks": len(seeds),
        "train_seed_tasks": len(train_seeds),
        "accepted_instructions": len(instructions),
        "rejected_instructions": len(rejected),
        "instruction_rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "classification_tasks": sum(r["is_classification"] for r in classifications),
        "generation_tasks": sum(not r["is_classification"] for r in classifications),
        "generated_instances": len(generated),
        "instance_rejections": len(rejected_instances),
        "splits": {
            name: {
                "instances": len(rows),
                "tasks": len({r["task_id"] for r in rows}),
                "generated_instances": sum(r["source"] == "generated" for r in rows),
            }
            for name, rows in splits.items()
        },
        "task_overlap": 0,
    }
    write_json(directory / "generation_summary.json", summary)
    event(
        directory,
        "dataset",
        f"Prepared train/valid/test: {len(splits['train'])}/{len(splits['valid'])}/{len(splits['test'])} instances; task overlap=0",
    )
    return summary


def artifact_inventory(directory):
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", ".run.lock", "events.jsonl"}
        and not path.name.endswith("_adapters.safetensors")
        and "requests" not in path.relative_to(directory).parts
    }


def verify_inventory(directory, inventory):
    for name, digest in inventory.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Artifact missing or changed: {name}")


def run(directory, config, *, train=True, replay_from=None):
    config.validate()
    directory = Path(directory)
    if not Path(config.seed_tasks).is_file():
        raise FileNotFoundError(
            f"Seed file not found: {config.seed_tasks}. Run from the repository root."
        )
    if replay_from and train:
        raise ValueError("Replay reproduces data processing only; pass --no-train")
    identity = {
        "schema_version": 1,
        "config": asdict(config),
        "train": train,
        "mode": "replay" if replay_from else "mlx",
        "seed_sha256": file_sha256(config.seed_tasks),
        "replay_sha256": file_sha256(replay_from) if replay_from else None,
    }
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity:
                raise ValueError(
                    "Run configuration/input changed. Choose a new --output directory."
                )
            if manifest["status"] == "complete":
                verify_inventory(directory, manifest["artifacts"])
                print(f"Verified completed run: {directory}")
                return manifest
        else:
            if any(p.name != ".run.lock" for p in directory.iterdir()):
                raise ValueError("Output directory must be empty or contain a matching manifest")
            manifest = {
                "identity": identity,
                "created_at": utc_now(),
                "environment": environment(),
                "source_sha256": {
                    p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
                },
            }
        manifest.update(status="running", last_started_at=utc_now())
        write_json(manifest_path, manifest)
        write_json(directory / "config.json", asdict(config))
        backend = None if replay_from else MLXBackend(config.model, config.revision)
        store = RequestStore(directory, backend, config.model, config.revision, replay_from)
        start = time.perf_counter()
        try:
            generate_data(directory, config, store)
            if backend:
                backend.close()
            if train:
                from .training import train_and_evaluate

                train_and_evaluate(directory, config)
            from .reporting import analyze

            analyze(directory)
            manifest.update(
                status="complete",
                completed_at=utc_now(),
                last_attempt_seconds=time.perf_counter() - start,
                artifacts=artifact_inventory(directory),
            )
            event(directory, "complete", f"Artifacts saved to {directory}")
        except BaseException as error:
            manifest.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {error}",
                stopped_at=utc_now(),
            )
            raise
        finally:
            if backend:
                backend.close()
            write_json(manifest_path, manifest)
        return manifest
