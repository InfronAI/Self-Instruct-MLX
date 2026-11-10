import argparse
import json
import os
import random
import re
import string

import tqdm

from self_instruct.filtering import NoveltyIndex
from self_instruct.gpt3_api import make_requests as make_gpt3_requests

random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx + 1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """Sample n machine instructions from a list of machine instructions."""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    # Chat models may include the first list number rather than continuing after it.
    raw_instructions = re.split(r"(?:^|\n)\s*\d+\s*[.)]\s*", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(
            find_word_in_string(word, inst)
            for word in [
                "image",
                "images",
                "graph",
                "graphs",
                "picture",
                "pictures",
                "file",
                "files",
                "map",
                "maps",
                "draw",
                "plot",
                "go to",
            ]
        ):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.lower().startswith("write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_requests",
        type=int,
        default=10000,
        help="Bound retries when the model produces no useful instructions.",
    )
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine", type=str, default=os.environ.get("OPENAI_MODEL"), help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used.",
    )
    args = parser.parse_args()
    if not args.engine:
        parser.error("Set --engine or OPENAI_MODEL to a supported Chat Completions model")
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(line) for line in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    if args.num_prompt_instructions > len(seed_instructions):
        raise ValueError("num_prompt_instructions exceeds the number of seed tasks")
    if min(args.request_batch_size, args.max_requests, args.num_prompt_instructions) < 1:
        raise ValueError("Batch size, max_requests and prompt count must be positive")
    novelty = NoveltyIndex(seed_instructions + machine_instructions)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            if request_idx >= args.max_requests:
                raise RuntimeError(
                    "Instruction generation budget exhausted; inspect saved responses"
                )
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, similarities=None, n=2
                )
                # sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions, args.num_prompt_instructions - len(prompt_instructions)
                )
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(
                    prompt_instructions, classification=args.use_clf_seed_tasks_only
                )
                batch_inputs.append(prompt)
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                if len(machine_instructions) >= args.num_instructions_to_generate:
                    break
                scores = novelty.compare(inst)
                if scores["max_similarity"] > 0.7:
                    continue
                machine_instructions.append(inst)
                novelty.add(inst)
                fout.write(
                    json.dumps(
                        {
                            "instruction": inst,
                            "most_similar": scores["most_similar"],
                            "avg_similarity_score": scores["avg_similarity_score"],
                            "metadata": metadata,
                            "request_idx": request_idx,
                        }
                    )
                    + "\n"
                )
                fout.flush()
                progress_bar.update(1)
            request_idx += 1
