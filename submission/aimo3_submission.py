"""
AIMO 3 Submission - SC-TIR (Self-Consistency with Tool-Integrated Reasoning)
Based on NuminaMath winning solution, adapted for AIMO 3.

Key changes from AIMO 1:
- Answers are 0-99999 (not mod 1000)
- Uses kaggle_evaluation API (not aimo)
- H100 GPUs available (can run larger models)
"""

import os
import re
import signal
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

# Configuration
@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"  # Default model

    # Decoding Parameters
    num_samples: int = 32          # Width: number of solution candidates
    num_generations: int = 4       # Depth: code execution rounds per candidate
    restart_on_fail: bool = True   # Retry if code block not generated

    # Sampling Parameters
    temperature: float = 0.7
    max_new_tokens: int = 2048

    # AIMO 3 specific
    answer_modulo: Optional[int] = None  # None = no modulo (0-99999 range)

    # Runtime
    is_submission: bool = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


# ============= Python REPL for Code Execution =============

class PythonREPL:
    """Sandboxed Python code executor with timeout."""

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(*_):
            raise TimeoutError(f"Timed out after {seconds} seconds.")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def __call__(self, query: str) -> tuple[bool, str]:
        # Add common imports
        imports = """import math
import numpy as np
import sympy as sp
from sympy import *
from fractions import Fraction
from itertools import permutations, combinations, product
from functools import reduce
"""
        query = imports + query

        # Ensure last line prints result
        query_lines = query.strip().split("\n")
        if "print(" not in query_lines[-1]:
            if "#" in query_lines[-1]:
                query_lines[-1] = query_lines[-1].split("#")[0]
            query_lines[-1] = "print(" + query_lines[-1] + ")"
        query = "\n".join(query_lines)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "solution.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(query)

            try:
                with self.time_limit(self.timeout):
                    result = subprocess.run(
                        ["python3", temp_file],
                        capture_output=True,
                        check=False,
                        text=True,
                        timeout=self.timeout,
                    )
                    if result.returncode == 0:
                        return True, result.stdout.strip()

                    # Parse error message
                    error_msg = result.stderr.strip()
                    lines = error_msg.split("\n")
                    filtered = []
                    for line in lines:
                        if "Traceback" in line or line == lines[-1]:
                            filtered.append(line)
                        elif temp_file in line:
                            filtered.append(line.replace(temp_file, "solution.py"))
                    return False, "\n".join(filtered)
            except TimeoutError as e:
                return False, str(e)
            except Exception as e:
                return False, str(e)


def execute_code_blocks(text: str, executor: PythonREPL) -> tuple[str, bool]:
    """Execute Python code blocks in the text."""
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)

    if not code_blocks:
        return "", False

    # Execute last code block
    code = code_blocks[-1]

    # Block dangerous operations
    for forbidden in ("subprocess", "os.system", "eval(", "exec(", "__import__"):
        if forbidden in code:
            return f"{forbidden} is not allowed", False

    try:
        success, output = executor(code)
        return output, success
    except Exception as e:
        return str(e), False


# ============= Answer Extraction =============

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} or \\fbox{}."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    # Find matching braces
    i = idx
    num_open = 0
    right_idx = None

    while i < len(text):
        if text[i] == "{":
            num_open += 1
        elif text[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_idx = i
                break
        i += 1

    if right_idx is None:
        return None

    # Remove \boxed{ prefix and } suffix
    boxed = text[idx:right_idx + 1]
    if boxed.startswith("\\boxed{"):
        return boxed[7:-1]
    elif boxed.startswith("\\fbox{"):
        return boxed[6:-1]
    return None


def normalize_answer(text: str) -> str:
    """Normalize answer string to extract numeric value."""
    if not text:
        return "-1"

    # Remove common units and formatting
    removals = [
        "square", "ways", "integers", "dollars", "mph", "inches", "ft",
        "hours", "km", "units", "points", "feet", "minutes", "digits",
        "cents", "degrees", "cm", "meters", "\\text{", "}", "\\mathrm{",
        "\\%", "%", "\\$", "$", "\\,", ",", " "
    ]

    result = text
    for r in removals:
        result = result.replace(r, "")

    # Handle fractions
    if "/" in result and not any(c.isalpha() for c in result):
        try:
            from fractions import Fraction
            frac = Fraction(result)
            if frac.denominator == 1:
                result = str(frac.numerator)
        except:
            pass

    return result.strip()


def parse_numeric_answer(text: str, modulo: Optional[int] = None) -> int:
    """Parse text to integer answer."""
    try:
        # Try direct float conversion
        value = float(text.replace(",", ""))
        value = round(value)

        # Apply modulo if specified
        if modulo:
            value = value % modulo

        # Ensure in valid range for AIMO 3
        if 0 <= value <= 99999:
            return int(value)
        elif value < 0:
            return 0
        else:
            return int(value) % 100000
    except:
        return -1


# ============= SC-TIR Algorithm =============

def generate_solution(problem: str, model, tokenizer, config: Config) -> list[str]:
    """Generate multiple solution candidates using SC-TIR."""

    # Format prompt
    prompt = f"""Solve this math problem step by step. Use Python code when calculations are needed.
Put your final numerical answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

    candidates = []
    executor = PythonREPL(timeout=5)

    for _ in range(config.num_samples):
        current_text = prompt

        for step in range(config.num_generations):
            # Generate next part
            inputs = tokenizer(current_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stop_strings=["```output"],
                    tokenizer=tokenizer,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated[len(current_text):]
            current_text = generated

            # Check for code blocks to execute
            if "```python" in new_text:
                output, success = execute_code_blocks(current_text, executor)
                if output:
                    current_text += f"\n```output\n{output}\n```\n"

            # Check if answer found
            if "\\boxed" in current_text:
                break

        candidates.append(current_text)

    return candidates


def extract_answers(candidates: list[str], modulo: Optional[int] = None) -> list[int]:
    """Extract numeric answers from solution candidates."""
    answers = []

    for text in candidates:
        boxed = extract_boxed_answer(text)
        if boxed:
            normalized = normalize_answer(boxed)
            value = parse_numeric_answer(normalized, modulo)
            if value >= 0:
                answers.append(value)

    return answers


def majority_vote(answers: list[int]) -> int:
    """Select answer by majority voting."""
    if not answers:
        return 0

    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def solve_problem(problem: str, model, tokenizer, config: Config) -> int:
    """Solve a single problem using SC-TIR."""

    # Generate candidates
    candidates = generate_solution(problem, model, tokenizer, config)

    # Extract answers
    answers = extract_answers(candidates, config.answer_modulo)

    # Majority vote
    if answers:
        return majority_vote(answers)

    return 0  # Default if no valid answers


# ============= Kaggle Submission Interface =============

def run_submission(config: Config):
    """Main submission loop for Kaggle."""

    print(f"=== AIMO 3 Submission ===")
    print(f"Model: {config.model_id}")
    print(f"Samples: {config.num_samples}, Generations: {config.num_generations}")
    print(f"Is submission: {config.is_submission}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model loaded.")

    if config.is_submission:
        # Kaggle submission mode
        import kaggle_evaluation.aimo_3_inference_server

        def predict(problem_df):
            problem = problem_df["problem"].iloc[0]
            answer = solve_problem(problem, model, tokenizer, config)
            return pd.DataFrame({"id": problem_df["id"], "answer": [answer]})

        server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        server.serve()
    else:
        # Local validation mode
        test_df = pd.read_csv("/Users/sabah/ai-content/AIMO/data/reference.csv")

        results = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            problem = row["problem"]
            true_answer = row["answer"]

            pred_answer = solve_problem(problem, model, tokenizer, config)

            results.append({
                "id": row["id"],
                "true": true_answer,
                "pred": pred_answer,
                "correct": true_answer == pred_answer
            })

            print(f"Problem {idx+1}: True={true_answer}, Pred={pred_answer}, Correct={true_answer == pred_answer}")

        results_df = pd.DataFrame(results)
        accuracy = results_df["correct"].mean()
        print(f"\n=== Results ===")
        print(f"Accuracy: {accuracy:.2%} ({results_df['correct'].sum()}/{len(results_df)})")


# ============= Main =============

if __name__ == "__main__":
    config = Config(
        model_id="Qwen/Qwen2.5-Math-7B-Instruct",
        num_samples=16,  # Reduce for testing
        num_generations=4,
        temperature=0.7,
        is_submission=False,
    )

    run_submission(config)
