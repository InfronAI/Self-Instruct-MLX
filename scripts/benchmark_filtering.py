"""Measure the original spawn-per-candidate ROUGE path against cached-token LCS."""

import argparse
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

from rouge_score import rouge_scorer

from self_instruct.artifacts import environment, read_jsonl, write_json
from self_instruct.filtering import NoveltyIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", default="experiments/m1-2026-11-10/instructions.jsonl")
    parser.add_argument("--output", default="experiments/filtering_benchmark.json")
    args = parser.parse_args()
    seeds = [r["instruction"] for r in read_jsonl("data/seed_tasks.jsonl")]
    candidates = [r["instruction"] for r in read_jsonl(args.instructions)][:12]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    timings, max_difference = [], 0
    # Two repetitions include fresh macOS spawn overhead exactly as in the original.
    for _ in range(2):
        started = time.perf_counter()
        original = []
        for candidate in candidates:
            with mp.get_context("spawn").Pool(4) as pool:
                original.append(
                    [
                        r["rougeL"].fmeasure
                        for r in pool.map(partial(scorer.score, candidate), seeds)
                    ]
                )
        original_seconds = time.perf_counter() - started
        started = time.perf_counter()
        index = NoveltyIndex(seeds)
        optimized = [index.compare(candidate) for candidate in candidates]
        optimized_seconds = time.perf_counter() - started
        for scores, result in zip(original, optimized):
            max_difference = max(
                max_difference,
                abs(max(scores) - result["max_similarity"]),
                abs(sum(scores) / len(scores) - result["avg_similarity_score"]),
            )
        timings.append(
            {
                "original_seconds": original_seconds,
                "optimized_seconds": optimized_seconds,
                "speedup": original_seconds / optimized_seconds,
            }
        )
    result = {
        "environment": environment(),
        "seed_instructions": len(seeds),
        "candidates": len(candidates),
        "workers_in_original": 4,
        "start_method": "spawn",
        "repetitions": timings,
        "max_score_difference": max_difference,
        "scope": "Small 175-seed pool; includes process startup and tokenization. "
        "Not an asymptotic or 52K-pool benchmark.",
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
