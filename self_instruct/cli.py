"""Public command line interface. Heavy MLX imports only occur when requested."""

import argparse
import json
import sys
from pathlib import Path

from .artifacts import environment
from .backends import MLXBackend
from .pipeline import RunConfig, run


def build_parser():
    parser = argparse.ArgumentParser(description="Self-Instruct on Apple Silicon")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Python, architecture and Metal")
    generate = commands.add_parser("run", help="Generate, filter, train and evaluate")
    generate.add_argument("--config", type=Path, default=Path("configs/m1_demo.json"))
    generate.add_argument("--output", type=Path, default=Path("runs/m1-demo"))
    generate.add_argument("--no-train", action="store_true", help="Stop after dataset preparation")
    generate.add_argument(
        "--replay-from", type=Path, help="Recorded requests.jsonl (requires --no-train)"
    )
    analysis = commands.add_parser("analyze", help="Regenerate Markdown/JSON analysis")
    analysis.add_argument("directory", type=Path)
    chat = commands.add_parser("chat", help="Run a real prompt with a saved adapter")
    chat.add_argument("--run", type=Path, required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-tokens", type=int, default=128)
    chat.add_argument("--base", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            info = environment()
            info["metal_available"] = False
            if info["system"] == "Darwin" and info["architecture"] == "arm64":
                try:
                    import mlx.core as mx

                    info["metal_available"] = mx.metal.is_available()
                except ImportError:
                    pass
            print(json.dumps(info, indent=2))
            if not info["metal_available"]:
                print(
                    "Local runtime unavailable. Install -e '.[mlx]' using native arm64 Python.",
                    file=sys.stderr,
                )
                return 1
        elif args.command == "run":
            config = RunConfig(**json.loads(args.config.read_text()))
            run(args.output, config, train=not args.no_train, replay_from=args.replay_from)
        elif args.command == "analyze":
            from .reporting import analyze

            analyze(args.directory)
            print(f"Analysis: {args.directory / 'analysis.md'}")
        elif args.command == "chat":
            config = RunConfig(**json.loads((args.run / "config.json").read_text()))
            adapter = None if args.base else str(args.run / "adapters")
            if adapter and not (Path(adapter) / "adapters.safetensors").is_file():
                raise ValueError("No completed adapter in this run")
            backend = MLXBackend(config.model, config.revision, adapter_path=adapter)
            response = backend.complete(
                [{"role": "user", "content": args.prompt}],
                max_tokens=args.max_tokens,
                temperature=0,
                seed=config.seed,
            )
            print(response["text"])
            print(
                f"\n[{response['generation_tps']:.1f} tokens/s; {response['generation_tokens']} tokens]",
                file=sys.stderr,
            )
            backend.close()
    except (ValueError, RuntimeError, OSError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(
            130, "Interrupted. Re-run the same command to reuse completed generation requests.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
