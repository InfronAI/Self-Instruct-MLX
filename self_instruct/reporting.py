"""Rebuild experiment analysis from recorded evidence, without an LLM."""

import json
from pathlib import Path
from statistics import mean

from .artifacts import atomic_text, read_jsonl, write_json


def analyze(directory):
    directory = Path(directory)
    summary_file = directory / "generation_summary.json"
    if not summary_file.exists():
        raise ValueError(f"No completed generation summary in {directory}")
    generation = json.loads(summary_file.read_text())
    requests = read_jsonl(directory / "requests.jsonl")
    responses = [r["response"] for r in requests]
    seconds = sum(r["elapsed_seconds"] for r in responses)
    tokens = sum(r["generation_tokens"] for r in responses)
    data = {
        "generation": generation,
        "inference": {
            "requests": len(requests),
            "output_tokens": tokens,
            "recorded_request_seconds": seconds,
            "output_tokens_per_wall_second": tokens / seconds if seconds else 0,
            "mean_decode_tokens_per_second": mean(r["generation_tps"] for r in responses),
            "peak_memory_gb": max(r["peak_memory_gb"] for r in responses),
        },
        "evaluation": None,
        "training": None,
    }
    if (directory / "evaluation" / "metrics.json").exists():
        data["evaluation"] = json.loads((directory / "evaluation" / "metrics.json").read_text())
    if (directory / "training_summary.json").exists():
        data["training"] = json.loads((directory / "training_summary.json").read_text())
    write_json(directory / "analysis.json", data)
    lines = [
        "# Experiment analysis",
        "",
        "Generated directly from saved artifacts; no model-based grading.",
        "",
        "## Dataset",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Accepted new instructions | {generation['accepted_instructions']} |",
        f"| Rejected instruction candidates | {generation['rejected_instructions']} |",
        f"| Classification / generation tasks | {generation['classification_tasks']} / {generation['generation_tasks']} |",
        f"| Retained generated instances | {generation['generated_instances']} |",
        f"| Cross-split identical task overlap | {generation['task_overlap']} |",
        "",
        "| Split | Tasks | Instances | Generated instances |",
        "|---|---:|---:|---:|",
    ]
    for name, counts in generation["splits"].items():
        lines.append(
            f"| {name} | {counts['tasks']} | {counts['instances']} | {counts['generated_instances']} |"
        )
    lines += [
        "",
        "## Local inference",
        "",
        f"- {len(requests)} generation/classification requests; {tokens:,} output tokens.",
        f"- Recorded inference time: {seconds:.2f} s (excludes initial model download/load).",
        f"- Effective output throughput including prompt processing: {tokens / seconds:.2f} tokens/s.",
        f"- Peak MLX allocated memory during generation: {data['inference']['peak_memory_gb']:.3f} GB.",
        "- Replay preserves original inference measurements; these are not replay execution timings.",
    ]
    if data["training"]:
        training = data["training"]
        lines += [
            "",
            "## LoRA training",
            "",
            f"- {training['iterations']} optimizer steps; {training['trainable_parameters']:,} trainable parameters.",
            f"- Training plus validation: {training['elapsed_seconds']:.2f} s; peak MLX memory: {training['peak_memory_gb']:.3f} GB.",
            "- Batch size 1; assistant-only loss; overlength examples dropped with row indices in `tokenization.json`.",
            "- Evaluation reloads the saved adapter from disk, rather than using the in-memory training model.",
        ]
    if data["evaluation"]:
        metrics = data["evaluation"]
        lines += [
            "",
            "## Paired held-out evaluation",
            "",
            "| Metric | Base | Adapted | Delta |",
            "|---|---:|---:|---:|",
        ]
        for key in ["test_loss", "perplexity", "rouge_l", "exact_match"]:
            lines.append(
                f"| {key} | {metrics['base'][key]:.6f} | {metrics['adapted'][key]:.6f} | {metrics['delta'][key]:+.6f} |"
            )
        lines += [
            "",
            f"Decoded examples: {metrics['base']['examples']}; loss examples: {metrics['base']['loss_examples']}.",
            f"Outputs reaching the token cap (base/adapted): {metrics['base']['truncated_outputs']}/{metrics['adapted']['truncated_outputs']}.",
            "",
            "This is a small pipeline validation, not a reproduction of the paper's accuracy gains. "
            "Lower loss/perplexity and higher ROUGE-L do not establish improved instruction following. "
            "Inspect both prediction files: wording differences, imperfect seed references, and truncated answers affect the metrics.",
        ]
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "The original seed tasks were split before generation. Only training seeds appear in bootstrapping prompts. "
        "All seed instructions participate in novelty filtering, including held-out instructions. "
        "Generated siblings stay in one split. Exact task overlap is checked; semantic or pretraining contamination is not ruled out.",
        "",
        "The demo uses an already instruction-tuned, quantized local model and a short LoRA run. "
        "The original experiment used GPT-3 and a much larger dataset. "
        "Synthetic examples can be wrong even when they pass schema, novelty and consistency filters. "
        "Memory figures are MLX allocator peaks, not total system RAM or process RSS.",
        "",
        "Reproduce: `python -m self_instruct analyze " + str(directory) + "`.",
        "",
    ]
    atomic_text(directory / "analysis.md", "\n".join(lines))
    try:
        plot_analysis(directory, data)
    except ImportError:
        pass  # Core analysis is independent of optional plotting dependencies.
    return data


def plot_analysis(directory, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "self-instruct"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    colors = ["#42b8a7", "#6977e8", "#f3b563"]
    counts = data["generation"]["splits"]
    axes[0].bar(list(counts), [x["instances"] for x in counts.values()], color=colors)
    axes[0].set_title("Task-disjoint data splits")
    axes[0].set_ylabel("Examples")
    logs = read_jsonl(directory / "training_metrics.jsonl")
    for kind, key, color in [
        ("train", "train_loss", colors[0]),
        ("validation", "val_loss", colors[1]),
    ]:
        rows = [r for r in logs if r["kind"] == kind]
        if rows:
            axes[1].plot(
                [r["iteration"] for r in rows],
                [r[key] for r in rows],
                marker="o",
                markersize=3,
                label=kind,
                color=color,
            )
    axes[1].set(
        title="Assistant-token loss", xlabel="Completed optimizer steps", ylabel="Cross entropy"
    )
    if logs:
        axes[1].legend(frameon=False)
    metrics = data["evaluation"]
    if metrics:
        axes[2].bar(
            ["Base", "LoRA"], [metrics[k]["rouge_l"] for k in ["base", "adapted"]], color=colors[:2]
        )
        axes[2].set_ylim(0, 1)
    axes[2].set_title("Held-out ROUGE-L (lexical proxy)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "analysis.png", dpi=160, metadata={"Software": "Self-Instruct"})
    fig.savefig(directory / "analysis.svg", metadata={"Date": None})
    plt.close(fig)
