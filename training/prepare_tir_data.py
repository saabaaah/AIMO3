"""
Prepare NuminaMath-TIR data for R1-Distill-32B LoRA fine-tuning.

Converts TIR traces into R1's chat format:
  User: problem + instruction
  Assistant: <think>reasoning with code</think>\n\nThe answer is \boxed{...}

Filters for quality: must have code block, output block, and \boxed{} answer.
Curates a 10K subset sorted by difficulty (shorter solutions = easier, train on mix).
"""
import json
import re
import random
from datasets import load_dataset

OUTPUT = "/home/ubuntu/AIMO3/training/r1_tir_train.jsonl"
TARGET_SIZE = 10000
SEED = 42

R1_USER_TEMPLATE = """{problem}

Solve this step-by-step. When you need to compute something, write Python code in ```python blocks. After each code block, I will run it and show you the output. Put your final integer answer in \\boxed{{}}."""


def has_quality(msg_content: str) -> bool:
    """Check if assistant message has code + output + boxed answer."""
    has_code = "```python" in msg_content
    has_output = "```output" in msg_content
    has_boxed = "\\boxed{" in msg_content or "\\boxed " in msg_content
    # Filter very short or very long
    if len(msg_content) < 200 or len(msg_content) > 12000:
        return False
    return has_code and has_output and has_boxed


def convert_to_r1_format(example: dict) -> dict | None:
    """Convert NuminaMath-TIR example to R1 training format."""
    msgs = example.get("messages", [])
    if len(msgs) < 2:
        return None

    problem = msgs[0]["content"]
    assistant = msgs[1]["content"]

    if not has_quality(assistant):
        return None

    # Wrap assistant response in <think>...</think> then final answer
    # Extract the \boxed{} answer
    boxed_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", assistant)
    if not boxed_match:
        return None
    answer = boxed_match.group(0)  # full \boxed{...}

    # Build R1-style response: thinking + answer
    # The thinking IS the TIR trace (reasoning + code + output)
    thinking = assistant.strip()

    r1_response = f"<think>\n{thinking}\n</think>\n\n{answer}"

    user_content = R1_USER_TEMPLATE.format(problem=problem)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": r1_response},
        ]
    }


def main():
    print("Loading NuminaMath-TIR from HF...")
    ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    print(f"Total examples: {len(ds)}")

    # Convert and filter
    converted = []
    for i, ex in enumerate(ds):
        result = convert_to_r1_format(ex)
        if result:
            result["_len"] = len(result["messages"][1]["content"])
            converted.append(result)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}, kept {len(converted)}")

    print(f"After quality filter: {len(converted)}")

    # Sort by length (difficulty proxy) and sample a balanced mix
    random.seed(SEED)
    converted.sort(key=lambda x: x["_len"])

    # Take samples from different difficulty tiers
    n = len(converted)
    easy = converted[:n // 3]
    medium = converted[n // 3:2 * n // 3]
    hard = converted[2 * n // 3:]

    per_tier = TARGET_SIZE // 3
    selected = (
        random.sample(easy, min(per_tier, len(easy)))
        + random.sample(medium, min(per_tier, len(medium)))
        + random.sample(hard, min(TARGET_SIZE - 2 * per_tier, len(hard)))
    )
    random.shuffle(selected)

    # Remove helper field
    for s in selected:
        del s["_len"]

    # Save as JSONL
    with open(OUTPUT, "w") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(selected)} examples to {OUTPUT}")
    # Stats
    avg_len = sum(len(s["messages"][1]["content"]) for s in selected) / len(selected)
    print(f"Avg assistant length: {avg_len:.0f} chars")


if __name__ == "__main__":
    main()
