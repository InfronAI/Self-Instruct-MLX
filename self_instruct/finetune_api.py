"""Optional cloud fine-tuning entry point using the current OpenAI SDK.

Running this command uploads the supplied training data and creates a paid
fine-tuning job. Local MLX runs never invoke it.
"""

import argparse
import json
import os
from pathlib import Path

from .artifacts import read_jsonl, write_json, write_jsonl


def convert_training_data(rows):
    converted = []
    for row in rows:
        if "messages" in row:
            messages = row["messages"]
            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
                or not all(isinstance(m.get("content"), str) for m in messages)
            ):
                raise ValueError("Expected text chat messages ending with an assistant answer")
            converted.append({"messages": messages})
        elif isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
            answer = row["completion"].removesuffix("<|endoftext|>").strip()
            if not answer:
                raise ValueError("Empty training completion")
            converted.append(
                {
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
        else:
            raise ValueError("Expected messages or prompt/completion fields")
    if not converted:
        raise ValueError("Training file is empty")
    return converted


def create_job(client, chat_file, model, suffix):
    with Path(chat_file).open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="fine-tune")
    return client.fine_tuning.jobs.create(
        model=model, training_file=uploaded.id, suffix=suffix, hyperparameters={"n_epochs": 2}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_file", type=Path)
    parser.add_argument("experiment_name")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    args = parser.parse_args()
    if not args.model:
        parser.error("Set OPENAI_MODEL or --model to a model that supports fine-tuning")
    from openai import OpenAI

    converted = convert_training_data(read_jsonl(args.training_file))
    output = args.training_file.with_suffix(".chat.jsonl")
    if output.exists():
        parser.error(f"Refusing to overwrite {output}; choose another input filename")
    write_jsonl(output, converted)
    with OpenAI(timeout=120, max_retries=0) as client:
        # Do not automatically retry job creation after ambiguous network failures.
        job = create_job(client, output, args.model, args.experiment_name)
    receipt = output.with_suffix(".job.json")
    write_json(receipt, job.model_dump())
    print(json.dumps({"job_id": job.id, "status": job.status, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
