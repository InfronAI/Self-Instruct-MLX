"""Exercise the real SDK against an in-memory HTTP transport. No API calls/cost."""

import json

import pytest

from self_instruct.gpt3_api import make_requests

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")


def call(client, **kwargs):
    arguments = dict(
        engine="test-chat-model",
        prompts=["first", "second"],
        max_tokens=64,
        temperature=0.7,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=1,
        stop_sequences=["END"],
        logprobs=1,
        n=2,
        best_of=1,
        retries=0,
        client=client,
    )
    arguments.update(kwargs)
    return make_requests(**arguments)


def reply(request, *, finish="stop", content="answer"):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": i,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content, "refusal": None},
                    "logprobs": {
                        "content": [
                            {
                                "token": "answer",
                                "bytes": [97],
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "answer", "logprob": -0.1, "bytes": [97]}
                                ],
                            }
                        ]
                    },
                }
                for i in range(body["n"])
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    )


def client_with(handler):
    return openai.OpenAI(
        api_key="test-key-not-a-secret",
        base_url="https://example.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_endpoint_and_batch_candidate_mapping():
    requests = []

    def handler(request):
        requests.append(request)
        content = json.loads(request.content)["messages"][-1]["content"] + " completion"
        return reply(request, content=content)

    with client_with(handler) as client:
        rows = call(client)
    assert len(requests) == 2 and all(r.url.path == "/v1/chat/completions" for r in requests)
    assert [r["response"]["choices"][0]["text"] for r in rows] == [
        "first completion",
        "second completion",
    ]
    assert all(len(r["response"]["choices"]) == 2 for r in rows)
    payload = json.loads(requests[0].content)
    assert payload["max_completion_tokens"] == 64
    assert payload["logprobs"] is True and payload["top_logprobs"] == 1
    assert "prompt" not in payload and "engine" not in payload and "best_of" not in payload
    assert rows[0]["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
    assert rows[0]["response"]["choices"][0]["logprobs"]["tokens"] == ["answer"]


def test_length_stop_survives_for_downstream_filters():
    from self_instruct.bootstrap_instructions import post_process_gpt3_response

    with client_with(lambda r: reply(r, finish="length")) as client:
        row = call(client, prompts="one", n=1)[0]
    assert row["response"]["choices"][0]["finish_reason"] == "length"
    assert post_process_gpt3_response(row["response"]) == []


def test_chat_numbered_instructions_preserve_first_item_and_case():
    from self_instruct.bootstrap_instructions import post_process_gpt3_response
    response = {"choices": [{"finish_reason": "stop", "text":
                "1. Explain how NASA and ESA cooperate.\n2. Describe a quiet morning in Paris."}]}
    assert post_process_gpt3_response(response) == [
        "Explain how NASA and ESA cooperate.", "Describe a quiet morning in Paris."]


def test_refusal_does_not_become_training_text():
    def handler(request):
        response = reply(request, content=None)
        body = response.json()
        body["choices"][0]["message"]["refusal"] = "Cannot comply"
        return httpx.Response(200, json=body)

    with client_with(handler) as client:
        result = call(client, prompts="one", n=1)[0]
    assert result["response"]["choices"][0]["text"] == ""
    assert result["response"]["choices"][0]["refusal"] == "Cannot comply"


def test_sdk_retries_rate_limit_and_does_not_retry_auth():
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                429, headers={"retry-after-ms": "1"}, json={"error": {"message": "rate limited"}}
            )
        return reply(request)

    with client_with(handler) as client:
        assert call(client, prompts="one", retries=1)
    assert len(attempts) == 2
    attempts.clear()

    def denied(request):
        attempts.append(request)
        return httpx.Response(401, json={"error": {"message": "invalid credentials"}})

    with client_with(denied) as client:
        with pytest.raises(RuntimeError, match="AuthenticationError"):
            call(client, prompts="one", retries=2)
    assert len(attempts) == 1


def test_best_of_and_invalid_inputs_fail_before_network():
    def forbidden(request):
        pytest.fail("No HTTP call expected")

    with client_with(forbidden) as client:
        with pytest.raises(ValueError, match="best_of"):
            call(client, best_of=3)
        with pytest.raises(TypeError):
            call(client, prompts=[1])
        with pytest.raises(ValueError, match="logprobs"):
            call(client, logprobs=30)
        assert call(client, prompts=[]) == []


def test_sampling_options_can_be_omitted_for_reasoning_models():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return reply(request)

    with client_with(handler) as client:
        call(
            client,
            prompts="one",
            temperature=None,
            top_p=None,
            logprobs=None,
            frequency_penalty=None,
            presence_penalty=None,
            stop_sequences=None,
        )
    assert all(
        key not in bodies[0]
        for key in [
            "temperature",
            "top_p",
            "logprobs",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        ]
    )


def test_plain_text_cli_and_bare_output_filename(tmp_path, monkeypatch):
    import runpy
    import sys

    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts.txt").write_text("first\\nline\nsecond\n")
    client = client_with(reply)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpt3_api",
            "--input_file",
            "prompts.txt",
            "--output_file",
            "responses.jsonl",
            "--engine",
            "test-chat-model",
        ],
    )
    # run_path executes the real CLI, including input decoding and durable JSONL export.
    import self_instruct.gpt3_api as module

    runpy.run_path(module.__file__, run_name="__main__")
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["first\nline", "second"]
    assert all(r["response"]["choices"][0]["text"] == "answer" for r in rows)


def test_cloud_finetuning_uses_chat_data_and_current_job_endpoint(tmp_path):
    from self_instruct.artifacts import write_jsonl
    from self_instruct.finetune_api import convert_training_data, create_job

    rows = convert_training_data([{"prompt": "Task", "completion": " answer<|endoftext|>"}])
    assert rows[0]["messages"][-1]["content"] == "answer"
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/v1/files":
            assert b'"messages"' in request.content
            return httpx.Response(
                200,
                json={
                    "id": "file-test",
                    "object": "file",
                    "bytes": 100,
                    "created_at": 1,
                    "filename": "train.jsonl",
                    "purpose": "fine-tune",
                },
            )
        assert request.url.path == "/v1/fine_tuning/jobs"
        body = json.loads(request.content)
        assert body["training_file"] == "file-test" and body["model"] == "test-ft-model"
        assert body["hyperparameters"]["n_epochs"] == 2
        return httpx.Response(200, json={"id": "ftjob-test", "status": "validating_files"})

    with client_with(handler) as client:
        job = create_job(client, path, "test-ft-model", "demo")
    assert job.id == "ftjob-test" and len(requests) == 2
    with pytest.raises(ValueError):
        convert_training_data([])
