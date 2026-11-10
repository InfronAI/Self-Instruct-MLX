# Experiment analysis

Generated directly from saved artifacts; no model-based grading.

## Dataset

| Measure | Value |
|---|---:|
| Accepted new instructions | 12 |
| Rejected instruction candidates | 1 |
| Classification / generation tasks | 7 / 5 |
| Retained generated instances | 19 |
| Cross-split identical task overlap | 0 |

| Split | Tasks | Instances | Generated instances |
|---|---:|---:|---:|
| train | 10 | 16 | 16 |
| valid | 18 | 19 | 3 |
| test | 8 | 8 | 0 |

## Local inference

- 27 generation/classification requests; 1,089 output tokens.
- Recorded inference time: 36.87 s (excludes initial model download/load).
- Effective output throughput including prompt processing: 29.54 tokens/s.
- Peak MLX allocated memory during generation: 3.002 GB.
- Replay preserves original inference measurements; these are not replay execution timings.

## LoRA training

- 20 optimizer steps; 327,680 trainable parameters.
- Training plus validation: 30.81 s; peak MLX memory: 2.925 GB.
- Batch size 1; assistant-only loss; overlength examples dropped with row indices in `tokenization.json`.
- Evaluation reloads the saved adapter from disk, rather than using the in-memory training model.

## Paired held-out evaluation

| Metric | Base | Adapted | Delta |
|---|---:|---:|---:|
| test_loss | 4.209108 | 3.321097 | -0.888011 |
| perplexity | 67.296476 | 27.690701 | -39.605775 |
| rouge_l | 0.193679 | 0.204799 | +0.011120 |
| exact_match | 0.000000 | 0.000000 | +0.000000 |

Decoded examples: 8; loss examples: 8.
Outputs reaching the token cap (base/adapted): 6/6.

This is a small pipeline validation, not a reproduction of the paper's accuracy gains. Lower loss/perplexity and higher ROUGE-L do not establish improved instruction following. Inspect both prediction files: wording differences, imperfect seed references, and truncated answers affect the metrics.

## Interpretation and limitations

The original seed tasks were split before generation. Only training seeds appear in bootstrapping prompts. All seed instructions participate in novelty filtering, including held-out instructions. Generated siblings stay in one split. Exact task overlap is checked; semantic or pretraining contamination is not ruled out.

The demo uses an already instruction-tuned, quantized local model and a short LoRA run. The original experiment used GPT-3 and a much larger dataset. Synthetic examples can be wrong even when they pass schema, novelty and consistency filters. Memory figures are MLX allocator peaks, not total system RAM or process RSS.

Reproduce: `python -m self_instruct analyze experiments/m1-final`.
