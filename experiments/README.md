# Experiment record / 实验记录

The authoritative result is **[m1-final](m1-final/)**, recorded on 2026-11-10 on an Apple M1 Pro with 16 GB memory. Its [manifest](m1-final/manifest.json) records configuration, seed hash, runtime, source hashes and artifact hashes. Model weights are pinned to revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.

最终结果以 **m1-final** 为准。原始数据、配置、模型版本、来源、训练与评估指标均保存在目录中；视频记录的是这个实验的实际执行。

| Artifact | Meaning |
|---|---|
| `m1-final/requests.jsonl` | Full generation/classification prompts and raw model responses, with timing and usage |
| `m1-final/instructions.jsonl` | 12 accepted tasks, round numbers, nearest neighbors and ROUGE-L scores |
| `m1-final/rejected_instructions.jsonl` | Candidate rejected for excessive task overlap |
| `m1-final/classifications.jsonl` | Model's classification/generation decisions |
| `m1-final/instances.jsonl` | 19 generated examples passing structural filters |
| `m1-final/seed_split.json` | Seed identities split before prompting: 151 train / 16 validation / 8 test |
| `m1-final/dataset/` | Task-disjoint chat data and provenance; training uses 16 generated examples, validation 16 human + 3 generated examples, test 8 human examples |
| `m1-final/tokenization.json` | Token-budget audit; no examples dropped in the final run |
| `m1-final/training_metrics.jsonl` | Per-step loss, token throughput, validation loss and MLX memory |
| `m1-final/adapters/` | Reloadable final LoRA adapter and configuration |
| `m1-final/evaluation/` | Both models' complete saved predictions, reference outputs and metrics |
| `m1-final/analysis.*` | Rebuildable report and charts |
| `filtering_benchmark.json` | Two timed original-vs-optimized ROUGE passes on 175 seeds and 12 candidates |
| `quality_review.jsonl` / `quality_review.md` | Inspection of all 19 generated examples and known semantic errors |
| `verification.json` | Final tests, fresh installation, replay and media checks |

## Interpretation

The 20-step run establishes that the model generates data, trains on its own generated examples, saves adapters, and reloads them for inference. The test set is small; six of eight base and adapted answers hit the 128-token limit. ROUGE-L changes and lower teacher-forced loss are not sufficient evidence of improved general instruction following. No downstream benchmark or human preference gain is claimed.

The two invalid synonym/antonym examples were **not removed after observing results**. They remain in the raw and exported data so the reported experiment is reproducible. The quality review explains why structurally valid synthetic data still needs semantic review.

The filtering benchmark reproduces the original `spawn` pool lifecycle. Its 85–96× speedup is dominated by avoiding repeated process startup on this small pool; the algorithm is still exact pairwise LCS. Candidate text comes from the first local pilot, whose input path is the benchmark script's default. Use `--instructions m1-final/instructions.jsonl` (from this directory with an adjusted seed path, or the root command in README) to benchmark a different sample; timings will differ.

## Diagnostic runs retained for transparency

These are development diagnostics, **not comparable final quality results**:

- `pilot-format-2026-11-10/`: the first 1.5B JSON trial exhausted its instruction budget. The model often returned an object array instead of the requested object containing strings. Responses document the format failures.
- `m1-2026-11-10/`: a 1.5B end-to-end pilot retained only eight generated examples. Some task types were wrong. It mixed seed examples into training, excluded two human references using the generated-data colon heuristic, and predates the loss-mask correction. Do not compare its loss values with the final run.
- `template_probe.jsonl`: a three-prompt local comparison with the original text templates. Classification worked, but examples sometimes omitted required delimiters or continued into unrelated tasks.
- `pilot-loss-2026-11-10/`: the 4B run before correcting MLX-LM 0.28.3's padding boundary in the loss.
- `pilot-template-2026-11-10/`: the 4B run before removing training-only empty `<think>` markup inserted by the model repository's completed-message template.

Final training uses the actual inference prefix + literal answer + EOS, and computes loss only on answer tokens with positions strictly below the unpadded length. A focused real-MLX regression test covers the padding boundary. Diagnostic weights and per-request cache copies are excluded from version control; their JSON data and reports are retained, and their manifests describe the original local artifacts rather than a portable completed run.

开发诊断目录保留了格式失败、小模型质量问题、损失边界与聊天模板问题的证据。它们不是最终效果报告，也不应与最终实验直接比较损失或质量。只有最终适配器作为可复用产物纳入版本管理。
