import json
from dataclasses import replace

import pytest

from self_instruct.artifacts import read_jsonl, run_lock, write_jsonl
from self_instruct.backends import RequestStore
from self_instruct.pipeline import RunConfig, run, split_seeds


class FakeBackend:
    """Explicit test double; never used by the public CLI or experiment results."""

    calls = 0
    fail_at = None

    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def complete(self, messages, **kwargs):
        type(self).calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("simulated interruption")
        prompt = messages[-1]["content"]
        if "Invent " in prompt:
            value = {
                "instructions": [
                    "Compose a gentle poem celebrating the first snowfall of winter.",
                    "Classify a customer review as positive or negative.",
                    "List three creative ways to reuse cardboard boxes at home.",
                    "Explain how tides influence the daily lives of coastal communities.",
                ]
            }
        elif "Is this a classification" in prompt:
            value = {"is_classification": "Task: Classify" in prompt}
        else:
            value = {
                "instances": [
                    {"input": "Example A", "output": "First useful answer."},
                    {"input": "Example B", "output": "Second useful answer."},
                ]
            }
        return {
            "text": json.dumps(value),
            "finish_reason": "stop",
            "prompt_tokens": 20,
            "generation_tokens": 30,
            "generation_tps": 10,
            "peak_memory_gb": 0,
            "elapsed_seconds": 3,
            "backend": "test-double",
        }


@pytest.fixture
def config(tmp_path, monkeypatch):
    import self_instruct.pipeline as pipeline
    import self_instruct.reporting as reporting

    FakeBackend.calls, FakeBackend.fail_at = 0, None
    monkeypatch.setattr(pipeline, "MLXBackend", FakeBackend)
    monkeypatch.setattr(reporting, "plot_analysis", lambda *a: None)
    seeds = read_jsonl("data/seed_tasks.jsonl")[:14]
    write_jsonl(tmp_path / "seeds.jsonl", seeds)
    return RunConfig(
        seed_tasks=str(tmp_path / "seeds.jsonl"),
        num_instructions=4,
        prompt_examples=3,
        eval_tasks=2,
        validation_tasks=2,
        max_rounds=2,
    )


def test_pipeline_resume_and_replay(tmp_path, config):
    output = tmp_path / "run"
    FakeBackend.fail_at = 4
    with pytest.raises(RuntimeError, match="simulated"):
        run(output, config, train=False)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    FakeBackend.fail_at = None
    receipt = run(output, config, train=False)
    assert receipt["status"] == "complete"
    # Exactly one interrupted request is retried; all completed requests are cached.
    assert FakeBackend.calls == 10
    calls = FakeBackend.calls
    run(output, config, train=False)
    assert calls == FakeBackend.calls
    replay = tmp_path / "replay"
    run(replay, config, train=False, replay_from=output / "requests.jsonl")
    assert calls == FakeBackend.calls
    for name in [
        "instructions.jsonl",
        "classifications.jsonl",
        "instances.jsonl",
        "dataset/train.jsonl",
        "dataset/valid.jsonl",
        "dataset/test.jsonl",
    ]:
        assert (replay / name).read_bytes() == (output / name).read_bytes()
    sets = [
        {r["task_id"] for r in read_jsonl(output / "dataset" / f"{s}_provenance.jsonl")}
        for s in ["train", "valid", "test"]
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    # Test and validation seed instructions must never occur in bootstrap examples.
    protected = [
        r["instruction"]
        for s in ["valid", "test"]
        for r in read_jsonl(output / "dataset" / f"{s}_provenance.jsonl")
        if r["source"] == "seed"
    ]
    for request in read_jsonl(output / "requests.jsonl"):
        if request["request"]["stage"] == "instructions":
            assert all(t not in request["request"]["messages"][-1]["content"] for t in protected)


def test_completed_run_guards(tmp_path, config):
    directory = tmp_path / "run"
    run(directory, config, train=False)
    with pytest.raises(ValueError, match="configuration"):
        run(directory, replace(config, seed=43), train=False)
    (directory / "instances.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        run(directory, config, train=False)


def test_request_replay_must_match(tmp_path):
    backend = FakeBackend()
    store = RequestStore(tmp_path / "a", backend, "model", "rev")
    store.request("stage", [{"role": "user", "content": "one"}])
    replay = RequestStore(tmp_path / "b", None, "model", "rev", tmp_path / "a/requests.jsonl")
    with pytest.raises(ValueError, match="No matching"):
        replay.request("stage", [{"role": "user", "content": "different"}])


def test_generation_budget_is_bounded(tmp_path, config, monkeypatch):
    original = FakeBackend.complete

    def truncated(self, messages, **kwargs):
        response = original(self, messages, **kwargs)
        response["finish_reason"] = "length"
        return response

    monkeypatch.setattr(FakeBackend, "complete", truncated)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        run(tmp_path / "run", config, train=False)
    assert FakeBackend.calls == config.max_rounds * config.max_attempts


def test_lock_prevents_two_writers(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass


def test_invalid_config_and_seed_partitions(config):
    with pytest.raises(ValueError):
        replace(config, max_rounds=0).validate()
    with pytest.raises(ValueError):
        replace(config, instances_per_instruction=6).validate()
    with pytest.raises(ValueError, match="Not enough"):
        split_seeds([], config)
    seeds = read_jsonl(config.seed_tasks)
    with pytest.raises(ValueError, match="unique"):
        split_seeds(seeds + seeds, config)


def test_replay_cannot_train(tmp_path, config):
    with pytest.raises(ValueError, match="no-train"):
        run(tmp_path / "out", config, replay_from=tmp_path / "absent")


def test_output_directory_is_not_overwritten(tmp_path, config):
    (tmp_path / "user.txt").write_text("keep me")
    with pytest.raises(ValueError, match="empty"):
        run(tmp_path, config, train=False)
    assert (tmp_path / "user.txt").read_text() == "keep me"
