"""Modern Chat Completions with the original Self-Instruct response contract.

The filename and make_requests positional arguments are kept for compatibility.
Every prompt is a separate chat request; candidates stay grouped by prompt.
"""

import argparse
import json
import os
import random
from datetime import datetime

import tqdm


def make_requests(
    engine,
    prompts,
    max_tokens,
    temperature,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop_sequences,
    logprobs,
    n,
    best_of,
    retries=3,
    api_key=None,
    organization=None,
    *,
    base_url=None,
    timeout=60.0,
    client=None,
):
    """Call /v1/chat/completions through OpenAI's current Python client.

    ``engine`` remains the positional model argument. ``max_tokens`` maps to
    ``max_completion_tokens``; integer logprobs maps to boolean + top_logprobs.
    Pass None for model-specific sampling options that the selected model does
    not support. SDK retry/timeout handling is bounded; exhausted errors raise
    rather than silently producing empty training data. An injected client is
    useful for an explicitly configured endpoint or network-free tests.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise TypeError("prompts must be a string or list of strings")
    if best_of not in (None, 1):
        raise ValueError("Chat Completions does not support best_of; use best_of=1")
    n = 1 if n is None else n
    if not engine:
        raise ValueError("Specify a current chat model with --engine or OPENAI_MODEL")
    if n < 1 or max_tokens < 1 or retries < 0 or timeout <= 0:
        raise ValueError("Require n/max_tokens/timeout > 0 and retries >= 0")
    if logprobs is not None and (type(logprobs) is not int or not 0 <= logprobs <= 20):
        raise ValueError("logprobs must be None or an integer from 0 to 20")
    if not prompts:
        return []
    try:
        from openai import OpenAI, OpenAIError
    except ImportError as error:
        raise RuntimeError("Install the API extra: pip install -e '.[api]'") from error
    own_client = client is None
    if own_client:
        options = {"max_retries": retries, "timeout": timeout}
        for name, value in [
            ("api_key", api_key),
            ("organization", organization),
            ("base_url", base_url),
        ]:
            if value is not None:
                options[name] = value
        client = OpenAI(**options)
    requester = client.with_options(max_retries=retries, timeout=timeout)
    parameters = {"model": engine, "max_completion_tokens": max_tokens, "n": n}
    for name, value in [
        ("temperature", temperature),
        ("top_p", top_p),
        ("frequency_penalty", frequency_penalty),
        ("presence_penalty", presence_penalty),
    ]:
        if value is not None:
            parameters[name] = value
    if stop_sequences:
        parameters["stop"] = stop_sequences
    if logprobs is not None:
        parameters.update(logprobs=True, top_logprobs=logprobs)
    results = []
    try:
        for prompt in prompts:
            response = requester.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Continue the supplied task prompt. "
                        "Return only the requested completion; do not repeat the prompt.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **parameters,
            )
            choices = []
            for choice in response.choices:
                tokens = choice.logprobs.content if choice.logprobs else None
                normalized_logprobs = (
                    None
                    if tokens is None
                    else {
                        "tokens": [t.token for t in tokens],
                        "token_logprobs": [t.logprob for t in tokens],
                        "top_logprobs": [
                            {p.token: p.logprob for p in t.top_logprobs} for t in tokens
                        ],
                    }
                )
                choices.append(
                    {
                        "text": choice.message.content or "",
                        "index": choice.index,
                        "finish_reason": choice.finish_reason,
                        "logprobs": normalized_logprobs,
                        "refusal": choice.message.refusal,
                    }
                )
            results.append(
                {
                    "prompt": prompt,
                    "response": {
                        "id": response.id,
                        "model": response.model,
                        "choices": choices,
                        "usage": response.usage.model_dump() if response.usage else None,
                        "system_fingerprint": response.system_fingerprint,
                    },
                    "created_at": datetime.now().astimezone().isoformat(),
                }
            )
    except OpenAIError as error:
        status = getattr(error, "status_code", None)
        raise RuntimeError(
            f"Chat Completions failed: {type(error).__name__} (status={status}). "
            "Check model access, parameters, endpoint and credentials."
        ) from error
    finally:
        if own_client:
            client.close()
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=os.environ.get("OPENAI_MODEL"),
        help="Chat Completions model (or OPENAI_MODEL).",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument("--logprobs", default=5, type=int, help="The `logprobs` parameter of GPT3")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="The `n` parameter of GPT3. The number of responses to generate.",
    )
    parser.add_argument(
        "--best_of",
        type=int,
        default=1,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server.",
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists.",
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time.",
    )
    args = parser.parse_args()
    if not args.engine:
        parser.error("Set --engine or OPENAI_MODEL to a supported Chat Completions model")
    return args


if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompts = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i : i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")
