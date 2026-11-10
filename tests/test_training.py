import pytest

from self_instruct.training import score_predictions, tokenized_dataset


class TemplateTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [20, 21]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Include a two-token assistant header, mirroring a real chat template.
        if add_generation_prompt:
            return [1, 2, 3, 10, 11]
        # A defective template invents reasoning tokens for a completed turn.
        # They must never enter the assistant-only supervised target.
        return [1, 2, 3, 10, 11, 70, 71, 20, 21, 99]


def test_target_mask_excludes_assistant_header():
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    data, rejected = tokenized_dataset(rows, TemplateTokenizer(), max_seq_length=32)
    assert rejected == []
    assert data[0][0][data[0][1] :] == [20, 21, 99]
    with pytest.raises(ValueError, match="No examples fit"):
        tokenized_dataset(rows, TemplateTokenizer(), max_seq_length=6)


def test_metrics_do_not_inflate_empty_or_truncated_predictions():
    rows = [
        {"prediction": "", "reference": "correct", "finish_reason": "stop"},
        {"prediction": "YES", "reference": "yes", "finish_reason": "length"},
    ]
    metrics = score_predictions(rows)
    assert metrics["rouge_l"] == 0.5
    assert metrics["exact_match"] == 0.5
    assert metrics["nonempty_rate"] == 0.5
    assert metrics["truncated_outputs"] == 1
    with pytest.raises(ValueError):
        score_predictions([])


def test_mlx_loss_excludes_padding_and_prompt_tokens():
    mx = pytest.importorskip("mlx.core")
    from self_instruct.training import assistant_loss

    class PerfectAnswerModel:
        def __call__(self, inputs):
            logits = mx.full((1, 5, 8), -20.0)
            # Target indices 2 and 3 are answer tokens 3 and 4. Other
            # positions predict wrong tokens, and must not affect the loss.
            logits[0, 1, 3] = 20
            logits[0, 2, 4] = 20
            logits[0, 0, 7] = 20
            logits[0, 3, 7] = 20
            logits[0, 4, 7] = 20
            return logits

    loss, count = assistant_loss(
        PerfectAnswerModel(), mx.array([[1, 2, 3, 4, 0, 0]]), mx.array([[2, 4]])
    )
    assert count.item() == 2
    assert loss.item() == pytest.approx(0, abs=1e-6)
