"""Lazy local inference and strict, content-addressed response replay."""

import gc
import json
import platform
import time
from pathlib import Path

from .artifacts import fingerprint, read_jsonl, utc_now, write_json, write_jsonl

DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


class MLXBackend:
    def __init__(self, model=DEFAULT_MODEL, revision=DEFAULT_REVISION, adapter_path=None):
        self.model_id, self.revision, self.adapter_path = model, revision, adapter_path
        self.model = self.tokenizer = None
        self.model_path = None

    def load(self):
        if self.model is not None:
            return
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError(
                "MLX requires native arm64 Python on an Apple Silicon Mac. "
                "Use replay for a hardware-independent pipeline check."
            )
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from mlx_lm import load
        except ImportError as error:
            raise RuntimeError("Install the local runtime: pip install -e '.[mlx]'") from error
        if not mx.metal.is_available():
            raise RuntimeError("Metal is unavailable. Use native arm64 Python, outside Rosetta.")
        # Bound the allocator cache on 8/16 GB unified-memory machines.
        mx.set_cache_limit(256 * 1024 * 1024)
        self.model_path = (
            str(Path(self.model_id).resolve())
            if Path(self.model_id).is_dir()
            else snapshot_download(
                self.model_id,
                revision=self.revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                    "*.jinja",
                    "README.md",
                ],
            )
        )
        self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)

    def complete(self, messages, *, max_tokens=512, temperature=0.7, seed=42):
        self.load()
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if len(prompt) + max_tokens > 8192:
            raise ValueError("Request exceeds the local 8192-token context budget.")
        started, pieces, last = time.perf_counter(), [], None
        for chunk in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.9 if temperature else 1.0),
        ):
            pieces.append(chunk.text)
            last = chunk
        if last is None:
            raise RuntimeError("Model returned no generation.")
        return {
            "text": "".join(pieces),
            "finish_reason": last.finish_reason,
            "prompt_tokens": last.prompt_tokens,
            "generation_tokens": last.generation_tokens,
            "generation_tps": last.generation_tps,
            "peak_memory_gb": last.peak_memory,
            "elapsed_seconds": time.perf_counter() - started,
            "model": self.model_id,
            "revision": self.revision,
            "backend": "mlx",
        }

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


class RequestStore:
    """A response is committed atomically before downstream processing.

    Replay matches the entire request including the seed and model revision;
    a changed prompt fails instead of silently substituting unrelated output.
    """

    def __init__(self, directory, backend, model, revision, replay_from=None):
        self.directory = Path(directory)
        self.backend, self.model, self.revision = backend, model, revision
        self.cache = self.directory / "requests"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.replay = (
            None if replay_from is None else {row["key"]: row for row in read_jsonl(replay_from)}
        )
        if replay_from is not None and not self.replay:
            raise ValueError(f"Replay archive is empty or missing: {replay_from}")
        self.used = {}

    def request(self, stage, messages, *, max_tokens=512, temperature=0.7, seed=42):
        spec = {
            "stage": stage,
            "model": self.model,
            "revision": self.revision,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        key = fingerprint(spec)
        path = self.cache / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
        elif self.replay is not None:
            if key not in self.replay:
                raise ValueError(f"No matching recorded response for {stage}: {key}")
            record = self.replay[key]
            write_json(path, record)
        else:
            record = {
                "key": key,
                "request": spec,
                "created_at": utc_now(),
                "response": self.backend.complete(
                    messages, max_tokens=max_tokens, temperature=temperature, seed=seed
                ),
            }
            write_json(path, record)
        if record.get("key") != key or record.get("request") != spec:
            raise ValueError(f"Request cache integrity mismatch: {path}")
        self.used[key] = record
        # Also retain a portable archive after every successful request.
        write_jsonl(self.directory / "requests.jsonl", self.used.values())
        return record
