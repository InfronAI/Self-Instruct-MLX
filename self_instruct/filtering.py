"""Self-Instruct filtering, with cached tokens and bounded LCS memory."""

import re
import string

from rouge_score import tokenizers

TOKENIZER = tokenizers.DefaultTokenizer(use_stemmer=False)
UNSUPPORTED = re.compile(
    r"\b(image|images|graph|graphs|picture|pictures|file|files|map|maps|draw|plot|go to)\b",
    re.IGNORECASE,
)


def instruction_problem(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        return "empty"
    if not 4 <= len(instruction.split()) <= 150:
        return "word_count"
    if UNSUPPORTED.search(instruction):
        return "unsupported_modality"
    if instruction.lower().startswith("write a program"):
        return "program_template"
    if instruction[0] in string.punctuation or not instruction[0].isascii():
        return "invalid_prefix"
    return None


def rouge_l_tokens(left, right):
    """Exactly ROUGE-L F1 (no stemming), O(min(m,n)) working memory."""
    if not left or not right:
        return 0.0
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for a in left:
        row = [0]
        for j, b in enumerate(right):
            row.append(previous[j] + 1 if a == b else max(previous[j + 1], row[-1]))
        previous = row
    return 2 * previous[-1] / (len(left) + len(right))


class NoveltyIndex:
    def __init__(self, instructions=()):
        self.instructions = list(instructions)
        self.tokens = [TOKENIZER.tokenize(value) for value in self.instructions]

    def add(self, instruction):
        self.instructions.append(instruction)
        self.tokens.append(TOKENIZER.tokenize(instruction))

    def compare(self, instruction):
        tokens = TOKENIZER.tokenize(instruction)
        scores = [rouge_l_tokens(tokens, other) for other in self.tokens]
        top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:10]
        return {
            "max_similarity": max(scores, default=0.0),
            "avg_similarity_score": sum(scores) / len(scores) if scores else 0.0,
            "most_similar": {self.instructions[i]: scores[i] for i in top},
        }


def clean_instances(instruction, instances, limit=5):
    """Reject inconsistent tasks; preserve order and reject malformed/empty outputs."""
    valid, rejected, seen, inputs = [], [], set(), {}
    for instance in instances:
        if not isinstance(instance, dict) or not all(
            isinstance(instance.get(key), str) for key in ("input", "output")
        ):
            rejected.append({"reason": "schema", "instance": instance})
            continue
        value = {
            "instruction": instruction,
            "input": instance["input"].strip(),
            "output": instance["output"].strip(),
        }
        inp, out = value["input"], value["output"]
        if inp and inp in inputs and inputs[inp] != out:
            return [], rejected + [{"reason": "conflicting_outputs", "input": inp}]
        inputs[inp] = out
        if not out or inp == out or inp.endswith(":") or out.endswith(":"):
            rejected.append({"reason": "invalid_instance", "instance": value})
        elif (inp, out) in seen:
            rejected.append({"reason": "duplicate_instance", "instance": value})
        else:
            seen.add((inp, out))
            valid.append(value)
    return valid[:limit], rejected
