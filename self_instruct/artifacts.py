"""Durable, human-readable experiment artifacts; no model imports."""

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = []
        for i, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{i}: {error}") from error
        return rows


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def event(directory, stage, message, **data):
    row = {"time": utc_now(), "stage": stage, "message": message, **data}
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    print(f"[{stage}] {message}", flush=True)


@contextmanager
def run_lock(directory):
    # The kernel releases this advisory lock after a crash; no stale PID-file problem.
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment():
    packages = {}
    for name in ["mlx", "mlx-lm", "transformers", "huggingface-hub", "numpy", "rouge-score"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    result = {
        "python": platform.python_version(),
        "system": platform.system(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "architecture": platform.machine(),
        "packages": packages,
    }
    if platform.system() == "Darwin":
        for field, key in [("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")]:
            result[field] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        result["memory_bytes"] = int(result["memory_bytes"])
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result
