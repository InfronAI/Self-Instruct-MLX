# Generated-data quality inspection / 生成数据质量复核

Scope: all **19 structurally accepted examples** in `m1-final/instances.jsonl`. Inspection was performed by the implementation agent against each task's requested input/output relationship; it is not an independent expert annotation or a formal accuracy benchmark. The per-row decisions are saved in `quality_review.jsonl` with zero-based row indices and task IDs.

- **17 examples appear plausible** under this limited inspection.
- **2 examples are invalid**: rows 6 and 7, task `generated_0004`, ask whether a word is a synonym, antonym or near-hyponym of a target. Neither input supplies a word/target pair. The assigned labels cannot be justified from the inputs.
- Task `generated_0009` substantially overlaps the emotion-classification task `generated_0006`. Lexical ROUGE-L filtering does not guarantee semantic diversity. “I won the lottery!” / “joy” also appears under both instructions.
- Classification tasks with more than two labels receive only two examples in this small run, so label coverage is incomplete.
- Open-ended tasks usually yielded one example although two were requested. The pipeline records actual counts instead of fabricating a second example.

The questionable examples remain in the saved training data. Changing them after training would make the report disagree with the recorded experiment. For a production corpus, review task completeness, label correctness, semantic redundancy and label coverage before scaling training.

在 19 条通过结构过滤的样本中，17 条经本次有限检查看起来合理，2 条同义词/反义词任务缺少必要的词对，因此标签无法成立。还有两项情绪分类任务语义相近，说明词面去重不能替代语义审核。本次是实施代理的检查，不是独立人工专家标注或正式准确率评测。原始问题样本未在实验完成后被悄悄删除。
