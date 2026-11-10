import random

import pytest
from rouge_score import rouge_scorer

from self_instruct.filtering import (
    TOKENIZER,
    NoveltyIndex,
    clean_instances,
    instruction_problem,
    rouge_l_tokens,
)
from self_instruct.prepare_for_finetuning import (
    filter_duplicate_instances,
    parse_instances_for_classification_task,
    parse_instances_for_generation_task,
)


def test_optimized_rouge_matches_reference():
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rng = random.Random(42)
    for _ in range(60):
        left = " ".join(
            rng.choices(
                ["AI", "science", "C++", "sea", "sea", "Don't", "2026"], k=rng.randrange(25)
            )
        )
        right = " ".join(
            rng.choices(
                ["AI", "C++", "sea", "science", "don't", "rain", "2026"], k=rng.randrange(25)
            )
        )
        assert rouge_l_tokens(TOKENIZER.tokenize(left), TOKENIZER.tokenize(right)) == pytest.approx(
            scorer.score(left, right)["rougeL"].fmeasure
        )


def test_novelty_keeps_acronyms_and_rejects_duplicates():
    instruction = "Explain the difference between NASA and ESA."
    index = NoveltyIndex([instruction])
    assert index.compare(instruction)["max_similarity"] == 1
    assert instruction in index.compare(instruction)["most_similar"]
    assert instruction_problem("Draw a picture of the sea.") == "unsupported_modality"
    assert instruction_problem("Write a program to sort these numbers.") == "program_template"


def test_conflicting_labels_drop_whole_task():
    rows, rejected = clean_instances(
        "Classify the sentence into two categories.",
        [{"input": "same", "output": "yes"}, {"input": "same", "output": "no"}],
    )
    assert rows == []
    assert rejected[-1]["reason"] == "conflicting_outputs"


def test_instance_filters_are_stable_and_validate_types():
    examples = [
        {"input": "x", "output": "A"},
        {"input": "y", "output": "B"},
        {"input": "x", "output": "A"},
        {"input": "z", "output": ""},
        {"input": 4, "output": "four"},
    ]
    rows, rejected = clean_instances("instruction", examples)
    assert [r["input"] for r in rows] == ["x", "y"]
    assert len(rejected) == 3
    assert filter_duplicate_instances([("t", "b", "1"), ("t", "a", "2"), ("t", "b", "1")]) == [
        ("t", "b", "1"),
        ("t", "a", "2"),
    ]


def test_original_parsers_drop_incomplete_last_example():
    metadata = {"response": {"choices": [{"finish_reason": "length"}]}}
    text = "Example 1\nInput: first\nOutput: one\nExample 2\nInput: second\nOutput: tw"
    assert parse_instances_for_generation_task(text, "task", metadata) == [("task", "first", "one")]
    text = "Class label: positive\nReview: nice\nClass label: negative\nReview: awfu"
    assert parse_instances_for_classification_task(text, "task", metadata) == [
        ("task", "Review: nice", "positive")
    ]
