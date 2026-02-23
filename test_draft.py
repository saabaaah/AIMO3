import os
import re
import json
import signal
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from mlx_lm import load as mlx_load, stream_generate
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

# ============= Config =============
@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    num_samples: int = 16       # independent attempts per problem (width)
    num_generations: int = 4    # code-execute loops per attempt (depth)
    temperature: float = 0.8
    max_new_tokens: int = 2048
    answer_modulo: Optional[int] = None
    results_dir: str = ".aimo3/results_7b_n16_g4"

# ============= Python REPL =============
class PythonREPL:
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

    def __call__(self, query: str) -> tuple:
        imports = (
            "import math\nimport numpy as np\nimport sympy as sp\n"
            "from sympy import *\nfrom fractions import Fraction\n"
            "from itertools import permutations, combinations, product\n"
            "from functools import reduce\n"
        )
        query = imports + query
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
                        capture_output=True, check=False, text=True,
                        timeout=self.timeout,
                    )
                    if result.returncode == 0:
                        return True, result.stdout.strip()
                    error_msg = result.stderr.strip()
                    lines = error_msg.split("\n")
                    filtered = [l for l in lines if "Traceback" in l or l == lines[-1]
                                or temp_file in l]
                    return False, "\n".join(filtered)
            except TimeoutError as e:
                return False, str(e)
            except Exception as e:
                return False, str(e)

def execute_code_blocks(text: str, executor: PythonREPL) -> tuple:
    TICK3 = chr(96) * 3
    PY_START = TICK3 + "python"
    PY_BLOCK_RE = re.compile(PY_START + r"(.*?)" + TICK3, re.DOTALL)

    code_blocks = PY_BLOCK_RE.findall(text)
    if not code_blocks:
        return "", False
    code = code_blocks[-1]
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
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i, num_open, right_idx = idx, 0, None
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
    boxed = text[idx:right_idx + 1]
    if boxed.startswith("\\boxed{"):
        return boxed[7:-1]
    elif boxed.startswith("\\fbox{"):
        return boxed[6:-1]
    return None

def normalize_answer(text: str) -> str:
    if not text:
        return "-1"
    removals = [
        "square", "ways", "integers", "dollars", "mph", "inches", "ft",
        "hours", "km", "units", "points", "feet", "minutes", "digits",
        "cents", "degrees", "cm", "meters", "\\text{", "}", "\\mathrm{",
        "\\%", "%", "\\$", "$", "\\,", ",", " "
    ]
    result = text
    for r in removals:
        result = result.replace(r, "")
    if "/" in result and not any(c.isalpha() for c in result):
        try:
            from fractions import Fraction
            frac = Fraction(result)
            if frac.denominator == 1:
                result = str(frac.numerator)
        except Exception:
            pass
    return result.strip()

def parse_numeric_answer(text: str, modulo: Optional[int] = None) -> int:
    try:
        value = round(float(text.replace(",", "")))
        if modulo:
            value = value % modulo
        if 0 <= value <= 99999:
            return int(value)
        elif value < 0:
            return 0
        else:
            return int(value) % 100000
    except Exception:
        return -1

# ============= Result Persistence =============
def save_result(results_dir: str, row_id, problem: str, predicted: int, gold: int, is_correct: bool):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    entry = {
        "id": str(row_id),
        "problem_preview": problem[:120],
        "predicted": predicted,
        "gold": gold,
        "correct": is_correct,
        "timestamp": datetime.now().isoformat(),
    }
    with open(Path(results_dir) / "results.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_done_ids(results_dir: str) -> set:
    results_file = Path(results_dir) / "results.jsonl"
    if not results_file.exists():
        return set()
    done = set()
    with open(results_file) as f:
        for line in f:
            try:
                done.add(json.loads(line.strip())["id"])
            except Exception:
                pass
    return done

def print_summary(results_dir: str):
    results_file = Path(results_dir) / "results.jsonl"
    if not results_file.exists():
        print("No results yet.")
        return
    results = []
    with open(results_file) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except Exception:
                pass
    correct = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\n=== Cumulative: {correct}/{total} = {correct/total:.1%} ===")
    for r in results:
        mark = "OK" if r["correct"] else "WRONG"
        print(f"  [{mark}] id={r['id']} predicted={r['predicted']} gold={r['gold']}")

# ============= SC-TIR Core (mlx-lm, sequential) =============
TICK3   = chr(96) * 3
OUT_START = TICK3 + "output"
PY_START  = TICK3 + "python"

def _generate_until_stop(model, tokenizer, prompt: str, max_tokens: int, sampler) -> str:
    """
    Stream tokens from mlx-lm and stop as soon as the ```output marker appears.
    This avoids wasting compute on tokens after the model signals it wants code run.
    """
    result = ""
    for response in stream_generate(model, tokenizer, prompt,
                                    max_tokens=max_tokens, sampler=sampler):
        result += response.text
        if OUT_START in result:
            # Trim to exactly the stop marker — discard anything after
            result = result[:result.find(OUT_START) + len(OUT_START)]
            break
    return result

def generate_solution(problem: str, model, tokenizer, config: Config) -> list:
    """
    SC-TIR: run num_samples independent attempts sequentially.
    Each attempt gets num_generations rounds of generate → execute → continue.

    Sequential (not batched) avoids the MPS memory fragmentation that killed
    the transformers version. mlx-lm frees memory cleanly after each generate call.
    """
    sampler  = make_sampler(temp=config.temperature)
    executor = PythonREPL(timeout=5)

    messages = [
        {"role": "system", "content": (
            "You are a helpful math assistant. Solve problems step by step. "
            f"Use Python code in {PY_START} blocks when calculations are needed. "
            "Put your final numerical answer in \\boxed{}."
        )},
        {"role": "user", "content": problem},
    ]
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    all_texts = []
    for sample_idx in range(config.num_samples):
        prompt    = base_prompt
        full_text = ""

        for step in range(config.num_generations):
            generated = _generate_until_stop(
                model, tokenizer, prompt, config.max_new_tokens, sampler
            )

            full_text += generated
            prompt    += generated

            # If the model wrote a code block, execute it and feed output back
            if PY_START in generated:
                code_output, _ = execute_code_blocks(full_text, executor)
                if code_output:
                    appendix   = f"\n{code_output}\n{TICK3}\n"
                    full_text += appendix
                    prompt    += appendix

            # Stop this attempt once a boxed answer appears
            if "\\boxed" in full_text:
                break

        all_texts.append(full_text)

    return all_texts

def extract_answers(candidates: list, modulo=None) -> list:
    answers = []
    for text in candidates:
        boxed = extract_boxed_answer(text)
        if boxed:
            normalized = normalize_answer(boxed)
            value = parse_numeric_answer(normalized, modulo)
            if value >= 0:
                answers.append(value)
    return answers

def majority_vote(answers: list) -> int:
    if not answers:
        return 0
    return Counter(answers).most_common(1)[0][0]

def solve_problem(problem: str, model, tokenizer, config: Config) -> int:
    candidates = generate_solution(problem, model, tokenizer, config)
    answers    = extract_answers(candidates, config.answer_modulo)
    if not answers:
        return 0
    winner, votes = Counter(answers).most_common(1)[0]
    print(f"  votes: {Counter(answers).most_common(5)}  → {winner} ({votes}/{len(answers)})")
    return winner

# ============= Load Model =============
config = Config(
    model_id="models/Qwen2.5-Math-7B-Instruct-4bit",
    num_samples=16,
    num_generations=4,
    temperature=0.8,
    max_new_tokens=1024,
    results_dir=".aimo3/results_7b_4bit_n16_g4_1k",
)

# How many problems to evaluate. Set to None to run all 90.
EVAL_LIMIT = 20

print("Loading model with mlx-lm...")
model, tokenizer = mlx_load(config.model_id)
print("Model loaded.")

# ============= Run Eval =============
ds = load_dataset("AI-MO/aimo-validation-aime", cache_dir="datasets/val-aime")

results_dir = config.results_dir
done_ids    = load_done_ids(results_dir)
if done_ids:
    print(f"Resuming — skipping {len(done_ids)} already-solved problems.")

correct = 0
total   = 0
for row in tqdm(ds["train"], desc="Evaluating"):
    if EVAL_LIMIT is not None and total >= EVAL_LIMIT:
        break
    row_id = str(row.get("id", row["problem"][:40]))
    if row_id in done_ids:
        continue

    predicted  = solve_problem(row["problem"], model, tokenizer, config)
    gold       = int(row["answer"])
    is_correct = (predicted == gold)
    correct   += is_correct
    total     += 1

    save_result(results_dir, row_id, row["problem"], predicted, gold, is_correct)
    print(f"[{'OK' if is_correct else 'WRONG'}] predicted={predicted} gold={gold}")

print(f"\nThis run: {correct}/{total} = {correct/total:.1%}" if total else "\nNo new problems solved.")
print_summary(results_dir)
