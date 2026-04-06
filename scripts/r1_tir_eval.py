"""
R1-TIR: DeepSeek-R1-Distill + Tool-Integrated Reasoning

Combines R1's long chain-of-thought with code execution.
- Generates long reasoning with ```python blocks
- Stops at code blocks, executes, injects output
- Continues reasoning with computation results
- Self-consistency voting across N candidates

This is the optimized Priority 1 pipeline for AIMO3.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import torch
from vllm import LLM, SamplingParams


# ============= Config =============

@dataclass
class Config:
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    num_samples: int = 32
    num_rounds: int = 5           # max code-exec rounds per candidate
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens_per_round: int = 16384
    max_model_len: int = 40960    # total context window
    code_timeout: int = 10
    gpu_mem_util: float = 0.92
    quantization: str | None = None


# ============= Code Execution =============

IMPORTS = """\
import math
import numpy as np
import sympy as sp
from sympy import *
from fractions import Fraction
from itertools import permutations, combinations, product
from functools import reduce
from collections import Counter, defaultdict
"""

FORBIDDEN = {"subprocess", "os.system", "__import__", "shutil", "pathlib"}
PY = sys.executable


def execute_code(code: str, timeout: int) -> tuple[bool, str]:
    for tok in FORBIDDEN:
        if tok in code:
            return False, f"{tok} is not allowed"
    src = IMPORTS + "\n" + code
    # Auto-print last expression if no print
    lines = src.strip().split("\n")
    last = lines[-1].strip()
    if last and not last.startswith(("print", "#", "import", "from", "def ", "class ", "if ", "for ", "while ", "try", "with ", "else", "elif", "except", "finally", "return", "raise", "assert", "pass", "break", "continue", "@")):
        if "=" not in last or last.startswith("print"):
            lines[-1] = f"print({last})"
        src = "\n".join(lines)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [PY, p], capture_output=True, text=True, timeout=timeout, check=False,
            )
            if r.returncode == 0:
                out = r.stdout.strip()
                return True, out[:1200] + ("... [truncated]" if len(out) > 1200 else "")
            err = r.stderr.strip().split("\n")
            return False, err[-1] if err else "unknown error"
        except subprocess.TimeoutExpired:
            return False, f"Timed out after {timeout}s"
        except Exception as e:
            return False, str(e)


def extract_last_code_block(text: str) -> str | None:
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
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
        j += 1
    return None


def to_int(s: str) -> int | None:
    if not s:
        return None
    s = s.strip()
    for r in ("\\text{", "\\mathrm{", "\\,", ",", " ", "$", "}", "\\"):
        s = s.replace(r, "")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return round(float(s))
    except ValueError:
        pass
    if "/" in s:
        try:
            from fractions import Fraction
            f = Fraction(s)
            if f.denominator == 1:
                return f.numerator
        except Exception:
            pass
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


# ============= Prompt =============

R1_USER_PROMPT = """\
{problem}

Solve this step-by-step. When you need to compute something, write Python code in ```python blocks. \
After each code block, I will run it and show you the output. \
Put your final integer answer in \\boxed{{}}.
"""

QWEN_MATH_USER_PROMPT = """\
{problem}

Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.
"""


def build_prompt(tokenizer, problem: str, is_r1: bool) -> str:
    if is_r1:
        messages = [
            {"role": "user", "content": R1_USER_PROMPT.format(problem=problem)},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a math expert solver."},
            {"role": "user", "content": QWEN_MATH_USER_PROMPT.format(problem=problem)},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ============= Core Loop =============

def solve_one(llm, tokenizer, problem: str, cfg: Config, is_r1: bool) -> tuple[int, dict]:
    base = build_prompt(tokenizer, problem, is_r1)
    N = cfg.num_samples

    suffixes = [""] * N
    done = [False] * N
    stats = {"code_ok": 0, "code_fail": 0, "rounds": 0}

    stops = ["```output", "```\n\n", "<|im_end|>", "<|endoftext|>"]
    if is_r1:
        stops.append("</think>")  # R1 ends thinking with </think>, then gives answer

    base_len = len(tokenizer.encode(base))
    max_prompt_tokens = cfg.max_model_len - 1024  # reserve for generation

    for rnd in range(cfg.num_rounds):
        stats["rounds"] = rnd + 1
        live = [i for i in range(N) if not done[i]]
        if not live:
            break

        # Check prompt lengths — force-stop candidates nearing context limit
        prompts = []
        batch_live = []
        for i in live:
            full = base + suffixes[i]
            prompt_len = base_len + len(tokenizer.encode(suffixes[i])) if suffixes[i] else base_len
            if prompt_len >= max_prompt_tokens:
                done[i] = True  # context full, stop this candidate
                continue
            prompts.append(full)
            batch_live.append(i)
        live = batch_live
        if not live:
            break
        sp = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens_per_round,
            stop=stops,
            include_stop_str_in_output=True,
            seed=rnd * 1000 + 42,
            n=1,
        )
        outs = llm.generate(prompts, sp, use_tqdm=False)

        for j, out in enumerate(outs):
            i = live[j]
            text = out.outputs[0].text
            suffixes[i] += text
            fr = out.outputs[0].finish_reason
            sr = str(out.outputs[0].stop_reason or "")

            # Check if done (has \boxed)
            if extract_boxed(suffixes[i]) is not None:
                done[i] = True
                continue

            # R1 ended thinking — now generate the answer part
            if sr == "</think>":
                # Continue to get the answer outside <think>
                continue

            # Code execution trigger
            triggered = sr in ("```output", "```\n\n") or (
                fr == "stop" and suffixes[i].rstrip().endswith("```")
                and suffixes[i].count("```python") > suffixes[i].count("```output")
            )
            if triggered:
                code = extract_last_code_block(suffixes[i])
                if code:
                    ok, res = execute_code(code, cfg.code_timeout)
                    stats["code_ok" if ok else "code_fail"] += 1
                    res = res[:600]  # cap output to prevent context bloat
                    if sr == "```output":
                        suffixes[i] += f"\n{res}\n```\n"
                    else:
                        sfx = "" if suffixes[i].endswith("\n") else "\n"
                        suffixes[i] += f"{sfx}```output\n{res}\n```\n"
                continue

            if fr == "length":
                continue  # keep going next round

            # EOS without \boxed — done
            if fr == "stop":
                done[i] = True

    # Force \boxed for unfinished candidates
    still_open = [i for i in range(N) if not done[i] and extract_boxed(suffixes[i]) is None]
    if still_open:
        force_prompts = [base + suffixes[i] + "\n\nThe final answer is \\boxed{" for i in still_open]
        sp = SamplingParams(temperature=0.0, max_tokens=64, stop=["}"], include_stop_str_in_output=True, n=1)
        outs = llm.generate(force_prompts, sp, use_tqdm=False)
        for j, out in enumerate(outs):
            suffixes[still_open[j]] += "\n\nThe final answer is \\boxed{" + out.outputs[0].text

    # Extract & vote
    answers = []
    for i in range(N):
        boxed = extract_boxed(suffixes[i])
        val = to_int(boxed) if boxed else None
        if val is not None and 0 <= val <= 99999:
            answers.append(val)

    stats["valid"] = len(answers)
    stats["unique"] = len(set(answers))
    if answers:
        top = Counter(answers).most_common(1)[0]
        stats["vote"] = top[1]
        stats["distribution"] = dict(Counter(answers).most_common(5))
        return top[0], stats
    return 0, stats


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-tok", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=40960)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--data", default="/home/ubuntu/AIMO3/data/reference.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    is_r1 = "r1" in args.model.lower() or "distill" in args.model.lower()

    cfg = Config(
        model_id=args.model,
        num_samples=args.samples,
        num_rounds=args.rounds,
        temperature=args.temp,
        max_tokens_per_round=args.max_tok,
        max_model_len=args.max_len,
        gpu_mem_util=args.gpu_mem,
        quantization=args.quant,
    )

    print(f"Loading {cfg.model_id} (is_r1={is_r1})...", flush=True)
    t0 = time.time()
    kw = dict(
        model=cfg.model_id,
        gpu_memory_utilization=cfg.gpu_mem_util,
        max_model_len=cfg.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    if cfg.quantization:
        kw["quantization"] = cfg.quantization
    else:
        kw["dtype"] = "bfloat16"
    llm = LLM(**kw)
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    results = []
    total_t0 = time.time()
    for idx, row in df.iterrows():
        t0 = time.time()
        pred, stats = solve_one(llm, tokenizer, row["problem"], cfg, is_r1)
        dt = time.time() - t0
        true_ans = int(row["answer"])
        ok = pred == true_ans
        print(
            f"[{idx+1}/{len(df)}] id={row['id']} true={true_ans} pred={pred} "
            f"{'OK' if ok else 'XX'} | {dt:.1f}s | "
            f"valid={stats['valid']}/{cfg.num_samples} vote={stats.get('vote',0)} "
            f"code_ok={stats['code_ok']} code_fail={stats['code_fail']} "
            f"top5={stats.get('distribution',{})}",
            flush=True,
        )
        results.append(dict(
            id=row["id"], true=true_ans, pred=pred, correct=ok,
            seconds=round(dt, 1), valid=stats["valid"],
            vote=stats.get("vote", 0), code_ok=stats["code_ok"],
            code_fail=stats["code_fail"], rounds=stats["rounds"],
        ))

    total_dt = time.time() - total_t0
    rdf = pd.DataFrame(results)
    rdf.to_csv(args.out, index=False)
    acc = rdf["correct"].mean()
    print(f"\n{'='*60}", flush=True)
    print(f"Accuracy: {rdf['correct'].sum()}/{len(rdf)} = {acc:.1%}", flush=True)
    print(f"Total: {total_dt:.1f}s, avg {total_dt/len(rdf):.1f}s/problem", flush=True)
    print(f"Saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
