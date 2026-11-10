"""Compact chat adaptations of the original few-shot task/instance prompts."""

import json


def instruction_messages(examples, count):
    return [
        {
            "role": "system",
            "content": "You create diverse instruction-following tasks. "
            "Return valid JSON only, with no markdown or explanations.",
        },
        {
            "role": "user",
            "content": f"Invent {count} NEW, diverse English text-only tasks inspired by the examples below. "
            "Each task must be a standalone instruction, 5-60 words. Do not solve it. "
            "Make half the tasks classification with explicit finite labels, and half open-ended. "
            "Every array element must be a single instruction string, with no other fields. "
            "Avoid repeating the example tasks, image tasks, external resources, or programming.\n"
            f"Examples: {json.dumps(examples, ensure_ascii=False)}\n"
            'The JSON schema is {"instructions": [string, string, string, string]}. '
            'Return one object with the key "instructions". Do not add task types or solutions.',
        },
    ]


def classification_messages(instruction):
    return [
        {
            "role": "user",
            "content": "Is this a classification task with a finite set of output labels?\n"
            'Example: "Classify a review as positive or negative." -> {"is_classification": true}\n'
            'Example: "Write a short poem about winter." -> {"is_classification": false}\n'
            'Example: "Rewrite the sentence politely." -> {"is_classification": false}\n'
            f"Task: {instruction}\n"
            'Return only a JSON object: {"is_classification": true} or {"is_classification": false}.',
        }
    ]


def instance_messages(instruction, classification, count):
    if classification:
        strategy = (
            "Use the OUTPUT-FIRST strategy: first choose different valid class labels "
            "from the task, then invent a short input for each label. "
            "Put output before input in each JSON object."
        )
        example = '{"instances": [{"output": "positive", "input": "I loved the concert."}]}'
    else:
        strategy = (
            "Use the INPUT-FIRST strategy: first invent a short input, then write the "
            "correct output for that input. Use an empty input if the task needs none. "
            "Keep each output under 70 words."
        )
        example = '{"instances": [{"input": "The meeting is canceled.", "output": "The meeting will not take place."}]}'
    return [
        {
            "role": "system",
            "content": "You create accurate instruction-tuning examples. "
            "Return valid JSON only, with no markdown.",
        },
        {
            "role": "user",
            "content": f"Generate {count} distinct examples that exactly satisfy this task:\n{instruction}\n"
            f"{strategy}\nReturn this structure (replace example values): {example}",
        },
    ]


def user_prompt(instruction, inp):
    return instruction.strip() + ("\n\nInput: " + inp.strip() if inp.strip() else "")


def chat_record(instance):
    return {
        "messages": [
            {"role": "user", "content": user_prompt(instance["instruction"], instance["input"])},
            {"role": "assistant", "content": instance["output"]},
        ]
    }
