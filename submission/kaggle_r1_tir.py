"""
AIMO 3 Kaggle Submission — R1-TIR Pipeline
DeepSeek-R1-Distill-Qwen-32B + Code Execution + Self-Consistency

Target: Kaggle H100 (80GB VRAM), 5-hour GPU budget, 50 problems.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch

IS_KAGGLE = os.path.exists("/kaggle")
IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

# ============= Adaptive Config =============

VRAM_GB = 0
if torch.cuda.is_available():
    VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9

@dataclass
class Config:
    # Model — upload R1-Distill-32B as Kaggle dataset or use HF hub
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # Auto-tune based on VRAM
    num_samples: int = 8 if VRAM_GB < 90 else 16
    num_rounds: int = 4
    max_tokens_per_round: int = 12288 if VRAM_GB < 90 else 16384
    max_model_len: int = 24576 if VRAM_GB < 90 else 32768
    gpu_mem_util: float = 0.90 if VRAM_GB < 90 else 0.92

    temperature: float = 0.6
    top_p: float = 0.95
    code_timeout: int = 10

CFG = Config()
print(f"VRAM: {VRAM_GB:.0f}GB | samples={CFG.num_samples} | max_tok={CFG.max_tokens_per_round} | max_len={CFG.max_model_len}")


# ============= Code Execution =============

IMPORTS = """\
import math, sys
import numpy as np
import sympy as sp
from sympy import *
from fractions import Fraction
from itertools import permutations, combinations, product
from functools import reduce
from collections import Counter, defaultdict
"""

FORBIDDEN = {"subprocess", "os.system", "__import__", "shutil", "pathlib"}


def execute_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    for tok in FORBIDDEN:
        if tok in code:
            return False, f"{tok} blocked"
    src = IMPORTS + "\n" + code
    lines = src.strip().split("\n")
    last = lines[-1].strip()
    skip = ("print", "#", "import", "from", "def ", "class ", "if ", "for ", "while ",
            "try", "with ", "else", "elif", "except", "finally", "return", "raise",
            "assert", "pass", "break", "continue", "@")
    if last and not last.startswith(skip) and "=" not in last:
        lines[-1] = f"print({last})"
        src = "\n".join(lines)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [sys.executable, p],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if r.returncode == 0:
                out = r.stdout.strip()
                return True, out[:1200] + ("..." if len(out) > 1200 else "")
            err = r.stderr.strip().split("\n")
            return False, err[-1] if err else "error"
        except subprocess.TimeoutExpired:
            return False, f"Timed out ({timeout}s)"
        except Exception as e:
            return False, str(e)


def last_code_block(text: str) -> str | None:
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


# ============= Answer Extraction =============

def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = i + 7
    depth, j = 1, start
    while j < len(text) and depth > 0:
        if text[j] == "{": depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0: return text[start:j]
        j += 1
    return None


def to_int(s: str) -> int | None:
    if not s: return None
    s = s.strip()
    for r in ("\\text{", "\\mathrm{", "\\,", ",", " ", "$", "}", "\\"):
        s = s.replace(r, "")
    try: return int(s)
    except ValueError: pass
    try: return round(float(s))
    except ValueError: pass
    if "/" in s:
        try:
            from fractions import Fraction
            f = Fraction(s)
            if f.denominator == 1: return f.numerator
        except: pass
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


# ============= Model Loading =============

from vllm import LLM, SamplingParams

def load_model():
    model_path = CFG.model_id
    # Check for local Kaggle dataset path
    if IS_KAGGLE:
        local = "/kaggle/input/r1-distill-qwen-32b"
        if os.path.exists(local):
            model_path = local
            print(f"Using local model: {local}")

    print(f"Loading {model_path}...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=CFG.gpu_mem_util,
        max_model_len=CFG.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print(f"Loaded in {time.time()-t0:.0f}s")
    return llm


# ============= R1-TIR Solver =============

R1_PROMPT = """\
{problem}

Solve this step-by-step. When you need to compute something, write Python code in ```python blocks. \
After each code block, I will run it and show you the output. \
Put your final integer answer in \\boxed{{}}.
"""

def build_prompt(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": R1_PROMPT.format(problem=problem)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def solve(llm, tokenizer, problem: str) -> int:
    base = build_prompt(tokenizer, problem)
    N = CFG.num_samples
    suffixes = [""] * N
    done = [False] * N

    stops = ["```output", "```\n\n", "<|im_end|>", "<|endoftext|>", "</think>"]

    base_len = len(tokenizer.encode(base))
    max_prompt_tokens = CFG.max_model_len - 1024

    for rnd in range(CFG.num_rounds):
        live = [i for i in range(N) if not done[i]]
        if not live:
            break

        # Skip candidates nearing context limit
        prompts, batch_live = [], []
        for i in live:
            plen = base_len + len(tokenizer.encode(suffixes[i])) if suffixes[i] else base_len
            if plen >= max_prompt_tokens:
                done[i] = True
                continue
            prompts.append(base + suffixes[i])
            batch_live.append(i)
        live = batch_live
        if not live:
            break
        sp = SamplingParams(
            temperature=CFG.temperature, top_p=CFG.top_p,
            max_tokens=CFG.max_tokens_per_round,
            stop=stops, include_stop_str_in_output=True,
            seed=rnd * 1000 + 42, n=1,
        )
        outs = llm.generate(prompts, sp, use_tqdm=False)

        for j, out in enumerate(outs):
            i = live[j]
            text = out.outputs[0].text
            suffixes[i] += text
            fr = out.outputs[0].finish_reason
            sr = str(out.outputs[0].stop_reason or "")

            if extract_boxed(suffixes[i]) is not None:
                done[i] = True
                continue

            if sr == "</think>":
                continue

            triggered = sr in ("```output", "```\n\n") or (
                fr == "stop" and suffixes[i].rstrip().endswith("```")
                and suffixes[i].count("```python") > suffixes[i].count("```output")
            )
            if triggered:
                code = last_code_block(suffixes[i])
                if code:
                    ok, res = execute_code(code, CFG.code_timeout)
                    res = res[:600]  # cap output length
                    if sr == "```output":
                        suffixes[i] += f"\n{res}\n```\n"
                    else:
                        sfx = "" if suffixes[i].endswith("\n") else "\n"
                        suffixes[i] += f"{sfx}```output\n{res}\n```\n"
                continue

            if fr == "length":
                continue

            if fr == "stop":
                done[i] = True

    # Force answer for unfinished
    still_open = [i for i in range(N) if not done[i] and extract_boxed(suffixes[i]) is None]
    if still_open:
        fps = [base + suffixes[i] + "\n\nThe final answer is \\boxed{" for i in still_open]
        sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["}"], include_stop_str_in_output=True, n=1)
        outs = llm.generate(fps, sp, use_tqdm=False)
        for j, out in enumerate(outs):
            suffixes[still_open[j]] += "\n\nThe final answer is \\boxed{" + out.outputs[0].text

    # Vote
    answers = []
    for i in range(N):
        boxed = extract_boxed(suffixes[i])
        val = to_int(boxed) if boxed else None
        if val is not None and 0 <= val <= 99999:
            answers.append(val)
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return 0


# ============= Main =============

def main():
    llm = load_model()
    tokenizer = llm.get_tokenizer()

    if IS_SUBMISSION:
        # Kaggle competition mode
        sys.path.insert(0, "/kaggle/input/ai-mathematical-olympiad-progress-prize-3")
        import kaggle_evaluation.aimo_3_inference_server

        def predict(df):
            problem = df["problem"].iloc[0]
            t0 = time.time()
            answer = solve(llm, tokenizer, problem)
            print(f"  Solved in {time.time()-t0:.0f}s -> {answer}", flush=True)
            return pd.DataFrame({"id": df["id"], "answer": [answer]})

        server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
        server.serve()
    else:
        # Local evaluation
        data_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/reference.csv"
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(__file__), "..", "data", "reference.csv")

        df = pd.read_csv(data_path)
        correct = 0
        t_total = time.time()
        for idx, row in df.iterrows():
            t0 = time.time()
            pred = solve(llm, tokenizer, row["problem"])
            dt = time.time() - t0
            ok = pred == int(row["answer"])
            correct += ok
            print(f"[{idx+1}/{len(df)}] true={row['answer']} pred={pred} {'OK' if ok else 'XX'} ({dt:.0f}s)", flush=True)

        total = time.time() - t_total
        print(f"\nAccuracy: {correct}/{len(df)} = {correct/len(df):.1%}")
        print(f"Total: {total:.0f}s ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
