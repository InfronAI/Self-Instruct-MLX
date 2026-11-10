"""Record real CLI subprocesses as asciicast, then render an MP4 and GIF preview.

No simulated output: the cast contains the subprocess output with original
timestamps. The video compresses pauses and shows the source elapsed time.
Requires the viz extra and ffmpeg. Run from the repository root.
"""

import argparse
import codecs
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from self_instruct.artifacts import file_sha256, utc_now, write_json


def record(cast, run_dir):
    if run_dir.exists():
        raise ValueError(f"For a fresh, real demo use an unused output directory: {run_dir}")
    commands = [
        [sys.executable, "-m", "self_instruct", "doctor"],
        [
            sys.executable,
            "-u",
            "-m",
            "self_instruct",
            "run",
            "--config",
            "configs/m1_demo.json",
            "--output",
            str(run_dir),
        ],
        [
            sys.executable,
            "-m",
            "self_instruct",
            "chat",
            "--run",
            str(run_dir),
            "--prompt",
            "Classify this review as positive, negative, or mixed. Return only the label. Review: The food was delicious but the service was painfully slow.",
            "--max-tokens",
            "64",
        ],
    ]
    started = time.monotonic()
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        TQDM_DISABLE="1",
        TOKENIZERS_PARALLELISM="false",
    )
    receipt = {
        "started_at": utc_now(),
        "commands": commands,
        "run": str(run_dir),
        "description": "Real local generation, LoRA training, held-out evaluation, and saved-adapter inference.",
    }
    with cast.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "version": 2,
                    "width": 110,
                    "height": 28,
                    "title": "Self-Instruct on M1 Pro",
                    "timestamp": int(time.time()),
                }
            )
            + "\n"
        )

        def emit(text):
            handle.write(
                json.dumps([round(time.monotonic() - started, 4), "o", text], ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            print(text, end="", flush=True)

        for command in commands:
            emit("\n$ " + shlex.join(["python", *command[1:]]) + "\n")
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            ) as process:
                while chunk := process.stdout.read1(4096):
                    emit(decoder.decode(chunk))
                emit(decoder.decode(b"", final=True))
                code = process.wait()
                emit(f"\n[exit {code}]\n")
                if code:
                    raise RuntimeError(f"Demo command failed with status {code}; inspect {cast}")
        emit(
            "\nDone. Raw data, adapter, predictions, metrics and charts are saved in "
            + str(run_dir)
            + ".\n"
        )
    receipt.update(
        elapsed_seconds=time.monotonic() - started, exit_code=0, cast_sha256=file_sha256(cast)
    )
    write_json(cast.with_suffix(".json"), receipt)


def render(cast, output):
    from PIL import Image, ImageDraw, ImageFont

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg to render the recording: brew install ffmpeg")
    data = [json.loads(s) for s in cast.read_text().splitlines()]
    events = data[1:]
    width, height, fps = 1440, 900, 10
    font_path = next(
        (
            p
            for p in [
                "/System/Library/Fonts/Menlo.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ]
            if Path(p).exists()
        ),
        None,
    )
    font = ImageFont.truetype(font_path, 19) if font_path else ImageFont.load_default(size=19)
    title_font = ImageFont.truetype(font_path, 26) if font_path else font
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    transcript = ""
    source_time, frames, poster = 0.0, 0, None
    with subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    ) as encoder:
        for i, (timestamp, _, text) in enumerate(events):
            transcript += ansi.sub("", text).replace("\r", "\n")
            source_time = timestamp
            lines = []
            for line in transcript.splitlines():
                if not line.strip() or line.startswith(("Fetching ", "Calculating loss")):
                    continue
                lines.extend(
                    textwrap.wrap(line, width=114, replace_whitespace=False, drop_whitespace=False)
                    or [""]
                )
            frame = Image.new("RGB", (width, height), "#0b1020")
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle(
                (28, 28, width - 28, height - 70),
                radius=22,
                fill="#131b2e",
                outline="#293750",
                width=2,
            )
            for x, color in [(60, "#f47777"), (87, "#efc267"), (114, "#63ceb0")]:
                draw.ellipse((x, 55, x + 14, 69), fill=color)
            draw.text(
                (155, 48),
                "SELF-INSTRUCT  /  M1 PRO  /  LOCAL END-TO-END",
                font=title_font,
                fill="#e0e8fa",
            )
            draw.line((50, 96, width - 50, 96), fill="#293750", width=2)
            for row, line in enumerate(lines[-25:]):
                color = "#75e0c0" if line.startswith(("$", "[complete]", "Done.")) else "#d4ddef"
                draw.text((52, 116 + row * 27), line, font=font, fill=color)
            draw.text(
                (45, height - 49),
                "REAL CLI RECORDING  |  pauses compressed  |  source elapsed "
                f"{int(source_time) // 60:02d}:{int(source_time) % 60:02d}",
                font=font,
                fill="#91a5c6",
            )
            next_time = events[i + 1][0] if i + 1 < len(events) else timestamp + 48
            hold = min(2.2, max(0.35, (next_time - timestamp) / 8))
            if text.startswith("\n$"):
                hold = 2.5
            if i + 1 == len(events):
                hold = 6
            for _ in range(max(1, round(hold * fps))):
                encoder.stdin.write(frame.tobytes())
                frames += 1
            if "[train] LoRA" in text or "[complete]" in text:
                poster = frame.copy()
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")
    (poster or frame).save(output.with_name("demo-poster.png"))
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-vf",
            "fps=3,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(output.with_suffix(".gif")),
        ],
        check=True,
    )
    write_json(
        output.with_name("video-metadata.json"),
        {
            "resolution": [width, height],
            "fps": fps,
            "duration_seconds": frames / fps,
            "source_duration_seconds": source_time,
            "cast_sha256": file_sha256(cast),
            "video_sha256": file_sha256(output),
            "timing": "Intervals divided by 8, clamped to 0.35–2.2 s; commands/final frame held longer.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("experiments/m1-final"))
    parser.add_argument("--cast", type=Path, default=Path("docs/assets/demo.cast"))
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.cast.parent.mkdir(parents=True, exist_ok=True)
    if not args.render_only:
        record(args.cast, args.run)
    render(args.cast, args.cast.with_suffix(".mp4"))


if __name__ == "__main__":
    main()
