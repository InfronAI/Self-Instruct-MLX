"""Memory-bounded MLX LoRA and paired evaluation on unseen seed tasks."""

import json
import math
import time
from pathlib import Path

from .artifacts import event, file_sha256, read_jsonl, write_json, write_jsonl
from .backends import MLXBackend
from .filtering import TOKENIZER, rouge_l_tokens
from .prompts import user_prompt


def score_predictions(predictions):
    if not predictions:
        raise ValueError("Cannot evaluate an empty test set")
    for row in predictions:
        pred, ref = TOKENIZER.tokenize(row["prediction"]), TOKENIZER.tokenize(row["reference"])
        row["rouge_l"] = rouge_l_tokens(pred, ref)
        row["exact_match"] = " ".join(row["prediction"].lower().split()) == " ".join(
            row["reference"].lower().split()
        )
    n = len(predictions)
    return {
        "examples": n,
        "rouge_l": sum(r["rouge_l"] for r in predictions) / n,
        "exact_match": sum(r["exact_match"] for r in predictions) / n,
        "nonempty_rate": sum(bool(r["prediction"].strip()) for r in predictions) / n,
        "truncated_outputs": sum(r["finish_reason"] == "length" for r in predictions),
    }


def predict_test(backend, directory, config, label):
    predictions = []
    for i, row in enumerate(read_jsonl(directory / "dataset" / "test_provenance.jsonl")):
        response = backend.complete(
            [{"role": "user", "content": user_prompt(row["instruction"], row["input"])}],
            max_tokens=config.eval_max_tokens,
            temperature=0,
            seed=config.seed + i,
        )
        predictions.append(
            {
                "task_id": row["task_id"],
                "instruction": row["instruction"],
                "input": row["input"],
                "reference": row["output"],
                "prediction": response.pop("text"),
                **response,
            }
        )
        event(directory, "evaluate", f"{label}: {i + 1} test examples decoded")
    metrics = score_predictions(predictions)
    write_jsonl(directory / "evaluation" / f"{label}_predictions.jsonl", predictions)
    return metrics


def tokenized_dataset(rows, tokenizer, max_seq_length):
    """Drop overlength examples explicitly; never silently truncate all target tokens.

    Use the *inference* chat prefix followed by the literal answer and EOS.
    Some Qwen community templates insert synthetic empty <think> blocks when
    formatting a completed assistant turn, even for non-thinking models. Those
    blocks are not model-generated answers and must not become training targets.
    """
    accepted, rejected = [], []
    for i, row in enumerate(rows):
        messages = row["messages"]
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        answer = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        if not answer or answer[-1] != tokenizer.eos_token_id:
            answer.append(tokenizer.eos_token_id)
        tokens = prefix + answer
        if len(tokens) > max_seq_length:
            rejected.append({"row": i, "reason": "sequence_too_long", "tokens": len(tokens)})
        elif len(tokens) <= len(prefix) + 1:
            rejected.append({"row": i, "reason": "no_answer_tokens", "tokens": len(tokens)})
        else:
            accepted.append((tokens, len(prefix)))
    if not accepted:
        raise ValueError("No examples fit max_seq_length; increase it or inspect the dataset")
    return accepted, rejected


def assistant_loss(model, batch, lengths):
    """Loss only on real answer tokens, including EOS and excluding padding.

    MLX-LM 0.28.3's default_loss uses positions <= length and can include one
    padded target. Target positions here are 1-based indices into the original
    sequence, so valid indices are strictly less than its token count.
    """
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits, targets)
    count = mask.sum()
    return (losses * mask).astype(mx.float32).sum() / count, count


def train_and_evaluate(directory, config):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    directory = Path(directory)
    receipt_path = directory / "training_summary.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            file_sha256(directory / "adapters" / "adapters.safetensors")
            != receipt["adapter_sha256"]
        ):
            raise ValueError("Saved adapter hash mismatch")
        if (directory / "evaluation" / "metrics.json").exists():
            event(directory, "train", "Reusing completed training and evaluation")
            return
    # An interrupted training stage restarts from the base model. It never pretends
    # that a weights-only checkpoint contains optimizer/RNG state.
    backend = MLXBackend(config.model, config.revision)
    backend.load()
    train_rows = read_jsonl(directory / "dataset" / "train.jsonl")
    valid_rows = read_jsonl(directory / "dataset" / "valid.jsonl")
    test_rows = read_jsonl(directory / "dataset" / "test.jsonl")
    datasets = {}
    length_audit = {}
    for name, rows in [("train", train_rows), ("valid", valid_rows), ("test", test_rows)]:
        datasets[name], rejected = tokenized_dataset(rows, backend.tokenizer, config.max_seq_length)
        length_audit[name] = {"accepted": len(datasets[name]), "rejected": rejected}
    write_json(directory / "tokenization.json", length_audit)
    event(directory, "evaluate", "Evaluating the base model on held-out tasks")
    np.random.seed(config.seed)
    before_loss = evaluate(
        backend.model,
        datasets["test"],
        batch_size=1,
        num_batches=-1,
        max_seq_length=config.max_seq_length,
        loss=assistant_loss,
    )
    before = predict_test(backend, directory, config, "base")
    before.update(
        test_loss=before_loss, perplexity=math.exp(before_loss), loss_examples=len(datasets["test"])
    )

    adapter_dir = directory / "adapters"
    adapter_dir.mkdir(exist_ok=True)
    lora = {
        "rank": config.lora_rank,
        "scale": 16.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    adapter_config = {
        "model": config.model,
        "revision": config.revision,
        "num_layers": config.lora_layers,
        "fine_tune_type": "lora",
        "lora_parameters": lora,
    }
    if config.lora_layers > len(backend.model.layers):
        raise ValueError("lora_layers exceeds the number of transformer layers")
    mx.random.seed(config.seed)
    np.random.seed(config.seed)
    backend.model.freeze()
    linear_to_lora_layers(backend.model, config.lora_layers, lora)
    trainable = sum(v.size for _, v in tree_flatten(backend.model.trainable_parameters()))
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    write_json(adapter_dir / "adapter_config.json", adapter_config)
    training_events = []

    class Recorder:
        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise RuntimeError("Non-finite training loss")
            training_events.append({"kind": "train", **info})
            write_jsonl(directory / "training_metrics.jsonl", training_events)

        def on_val_loss_report(self, info):
            if not math.isfinite(info["val_loss"]):
                raise RuntimeError("Non-finite validation loss")
            training_events.append({"kind": "validation", **info})
            write_jsonl(directory / "training_metrics.jsonl", training_events)

    event(
        directory,
        "train",
        f"LoRA: {config.train_iters} steps, batch=1, {trainable:,} trainable parameters",
    )
    started = time.perf_counter()
    mx.reset_peak_memory()
    train(
        backend.model,
        optim.Adam(learning_rate=config.learning_rate),
        datasets["train"],
        datasets["valid"],
        args=TrainingArgs(
            batch_size=1,
            iters=config.train_iters,
            val_batches=-1,
            steps_per_report=1,
            steps_per_eval=max(1, config.train_iters // 2),
            steps_per_save=config.train_iters,
            max_seq_length=config.max_seq_length,
            adapter_file=str(adapter_dir / "adapters.safetensors"),
            grad_checkpoint=True,
        ),
        training_callback=Recorder(),
        loss=assistant_loss,
    )
    train_seconds = time.perf_counter() - started
    peak = mx.get_peak_memory() / 1e9
    # Release the mutated model and reload the saved artifact for an actual deployment check.
    backend.close()
    backend = MLXBackend(config.model, config.revision, adapter_path=str(adapter_dir))
    backend.load()
    event(directory, "evaluate", "Reloaded saved LoRA adapter; evaluating the same held-out tasks")
    np.random.seed(config.seed)
    after_loss = evaluate(
        backend.model,
        datasets["test"],
        batch_size=1,
        num_batches=-1,
        max_seq_length=config.max_seq_length,
        loss=assistant_loss,
    )
    after = predict_test(backend, directory, config, "adapted")
    after.update(
        test_loss=after_loss, perplexity=math.exp(after_loss), loss_examples=len(datasets["test"])
    )
    backend.close()
    write_json(
        directory / "evaluation" / "metrics.json",
        {
            "base": before,
            "adapted": after,
            "delta": {
                key: after[key] - before[key]
                for key in ["test_loss", "perplexity", "rouge_l", "exact_match"]
            },
            "method": "Greedy decoding; identical held-out seed tasks and prompts; assistant-only token loss. "
            "ROUGE-L and exact match are lexical proxies, not human judgments. "
            "Loss excludes overlength examples; generation includes all held-out examples.",
        },
    )
    write_json(
        receipt_path,
        {
            "iterations": config.train_iters,
            "trainable_parameters": trainable,
            "elapsed_seconds": train_seconds,
            "peak_memory_gb": peak,
            "batch_size": 1,
            "max_seq_length": config.max_seq_length,
            "adapter_sha256": file_sha256(adapter_dir / "adapters.safetensors"),
            "adapter_reloaded": True,
        },
    )
