"""
SC-TIR evaluation on reference.csv using vLLM + batched code execution.

Pipeline per problem:
  1. Generate N candidates in parallel, stopping at "```output"
  2. For each candidate with a python code block, execute and append output
  3. Continue generation; repeat for M rounds
  4. Extract \\boxed{...} from each candidate, majority vote
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import torch
from vllm import LLM, SamplingParams


# ============= Config =============

@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    num_samples: int = 16          # candidates per problem
    num_generations: int = 3       # code-exec rounds per candidate
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens_per_round: int = 2048
    max_model_len: int = 4096
    code_timeout: int = 6
    gpu_mem_util: float = 0.85
    mode: str = "tir"              # "tir" or "cot"
    quantization: str | None = None  # e.g. "awq" or "awq_marlin"


# ============= Python REPL =============

IMPORTS = """import math
import numpy as np
import sympy as sp
from sympy import *
from fractions import Fraction
from itertools import permutations, combinations, product
from functools import reduce
"""

FORBIDDEN = ("subprocess", "os.system", "eval(", "exec(", "__import__", "open(")

PY_INTERP = sys.executable  # venv python


def execute_code(code: str, timeout: int) -> tuple[bool, str]:
    for tok in FORBIDDEN:
        if tok in code:
            return False, f"{tok} is not allowed"
    src = IMPORTS + code
    # Make sure last stmt prints
    lines = src.strip().split("\n")
    last = lines[-1].strip()
    if last and not last.startswith(("print", "#")) and "=" not in last:
        lines[-1] = f"print({last})"
        src = "\n".join(lines)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py")
        with open(p, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [PY_INTERP, p],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            if r.returncode == 0:
                out = r.stdout.strip()
                if len(out) > 800:
                    out = out[:800] + "... [truncated]"
                return True, out
            err = r.stderr.strip().split("\n")
            # Keep last line of traceback
            return False, err[-1] if err else "error"
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        except Exception as e:
            return False, str(e)


def last_code_block(text: str) -> str | None:
    """Return last ```python ... ``` block's code."""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


# ============= Answer extraction =============

def extract_boxed(text: str) -> str | None:
    """Balanced-brace \\boxed{...} extraction (last occurrence)."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = i + len("\\boxed{")
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


def to_int_answer(s: str) -> int | None:
    if s is None:
        return None
    # Strip formatting
    s = s.strip()
    for r in ("\\text{", "\\,", ",", " ", "$", "}"):
        s = s.replace(r, "")
    # Try int, float, fraction
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
    # Pull first integer from string
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


# ============= SC-TIR =============

# Qwen2.5-Math official TIR system prompt
SYSTEM = (
    "Please integrate natural language reasoning with programs to solve the problem "
    "above, and put your final answer within \\boxed{}."
)


def build_prompt(tokenizer, problem: str) -> str:
    # Qwen2.5-Math TIR convention: instruction sits in the user turn, right after the problem
    user = problem.strip() + "\n\n" + SYSTEM
    messages = [
        {"role": "system", "content": "You are a math expert solver."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def sc_tir(llm: LLM, tokenizer, problem: str, cfg: Config) -> tuple[int, dict]:
    """Run SC-TIR on a single problem, return (predicted_int, stats)."""
    base_prompt = build_prompt(tokenizer, problem)

    # Stop on: TIR output marker, OR closing triple-backtick right after a code block,
    # OR end-of-turn tokens.
    stops = ["```output", "```\n\n", "<|im_end|>", "<|endoftext|>"]
    # Each candidate has its own running prompt suffix
    suffixes = [""] * cfg.num_samples
    done = [False] * cfg.num_samples
    stats = {"rounds": 0, "code_ok": 0, "code_fail": 0, "timeouts": 0}

    for rnd in range(cfg.num_generations):
        stats["rounds"] = rnd + 1
        # Build prompts for live candidates
        live_idx = [i for i, d in enumerate(done) if not d]
        if not live_idx:
            break
        prompts = [base_prompt + suffixes[i] for i in live_idx]
        sp = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens_per_round,
            stop=stops,
            include_stop_str_in_output=True,
            seed=42 + rnd,  # vary per round
            n=1,
        )
        outputs = llm.generate(prompts, sp, use_tqdm=False)

        for idx_in_batch, out in enumerate(outputs):
            i = live_idx[idx_in_batch]
            text = out.outputs[0].text
            suffixes[i] += text
            finish_reason = out.outputs[0].finish_reason
            stop_reason = str(out.outputs[0].stop_reason or "")

            # If \boxed already appeared, mark done
            if extract_boxed(suffixes[i]) is not None:
                done[i] = True
                continue

            # If we stopped on "```output" or "```\n\n" (closing code block),
            # the model wants code execution. Extract last python block and run it.
            trigger_exec = stop_reason in ("```output", "```\n\n") or (
                finish_reason == "stop"
                and suffixes[i].rstrip().endswith("```")
                and suffixes[i].count("```python") > suffixes[i].count("```output")
            )
            if trigger_exec:
                code = last_code_block(suffixes[i])
                if code:
                    ok, res = execute_code(code, cfg.code_timeout)
                    if ok:
                        stats["code_ok"] += 1
                    else:
                        stats["code_fail"] += 1
                        if "timeout" in res:
                            stats["timeouts"] += 1
                    # Inject the output block so the model can reason about it
                    if stop_reason == "```output":
                        # already has "```output" at end from include_stop_str_in_output
                        suffixes[i] += f"\n{res}\n```\n"
                    else:
                        # model closed the code block without writing "```output"
                        if not suffixes[i].endswith("\n"):
                            suffixes[i] += "\n"
                        suffixes[i] += f"```output\n{res}\n```\n"
            elif finish_reason == "length":
                # Hit max_tokens mid-sentence — keep going next round
                pass
            elif finish_reason == "stop" and stop_reason in ("<|im_end|>", "<|endoftext|>"):
                # Natural EOS with no \boxed — done (even if wrong)
                done[i] = True

    # Final generation pass to force \boxed if any candidate is still open
    live_idx = [i for i, d in enumerate(done) if not d]
    if live_idx:
        prompts = [
            base_prompt + suffixes[i] + "\nFinal answer: \\boxed{"
            for i in live_idx
        ]
        sp = SamplingParams(
            temperature=0.0, max_tokens=32, stop=["}"],
            include_stop_str_in_output=True, n=1,
        )
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for k, out in enumerate(outs):
            i = live_idx[k]
            suffixes[i] += "\nFinal answer: \\boxed{" + out.outputs[0].text

    # Extract & vote
    cand_answers: list[int] = []
    for i in range(cfg.num_samples):
        boxed = extract_boxed(suffixes[i])
        ans = to_int_answer(boxed) if boxed else None
        if ans is not None and 0 <= ans <= 99999:
            cand_answers.append(ans)

    stats["valid_answers"] = len(cand_answers)
    stats["unique_answers"] = len(set(cand_answers))

    if cand_answers:
        top = Counter(cand_answers).most_common(1)[0]
        pred = top[0]
        stats["vote_count"] = top[1]
        stats["all_answers"] = dict(Counter(cand_answers))
    else:
        pred = 0
        stats["vote_count"] = 0
        stats["all_answers"] = {}

    return pred, stats


# ============= Main =============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tok", type=int, default=1024)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--data", default="/home/ubuntu/AIMO3/data/reference.csv")
    ap.add_argument("--out", default="/home/ubuntu/AIMO3/scripts/sc_tir_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="stop after N problems")
    ap.add_argument("--quant", default=None, help="e.g. awq, gptq, awq_marlin")
    args = ap.parse_args()

    cfg = Config(
        model_id=args.model,
        num_samples=args.samples,
        num_generations=args.rounds,
        temperature=args.temp,
        max_tokens_per_round=args.max_tok,
        max_model_len=args.max_len,
    )

    print(f"Loading {cfg.model_id} on {torch.cuda.get_device_name(0)}...")
    t0 = time.time()
    llm_kwargs = dict(
        model=cfg.model_id,
        gpu_memory_utilization=cfg.gpu_mem_util,
        max_model_len=cfg.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    if args.quant:
        llm_kwargs["quantization"] = args.quant
    else:
        llm_kwargs["dtype"] = "bfloat16"
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s")

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    results = []
    total_t0 = time.time()
    for i, row in df.iterrows():
        t0 = time.time()
        pred, stats = sc_tir(llm, tokenizer, row["problem"], cfg)
        dt = time.time() - t0
        true = int(row["answer"])
        correct = pred == true
        print(
            f"[{i+1}/{len(df)}] id={row['id']} true={true} pred={pred} "
            f"{'OK' if correct else 'XX'} | {dt:.1f}s | valid={stats['valid_answers']}/{cfg.num_samples} "
            f"unique={stats['unique_answers']} vote={stats['vote_count']} "
            f"code_ok={stats['code_ok']} code_fail={stats['code_fail']} "
            f"top3={dict(Counter(stats['all_answers']).most_common(3))}",
            flush=True,
        )
        results.append({
            "id": row["id"], "true": true, "pred": pred, "correct": correct,
            "seconds": round(dt, 1),
            "valid_answers": stats["valid_answers"],
            "unique_answers": stats["unique_answers"],
            "vote_count": stats["vote_count"],
            "code_ok": stats["code_ok"], "code_fail": stats["code_fail"],
            "rounds_used": stats["rounds"],
        })

    total_dt = time.time() - total_t0
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out, index=False)
    acc = res_df["correct"].mean()
    print(f"\n=== Summary ===")
    print(f"Accuracy: {res_df['correct'].sum()}/{len(res_df)} = {acc:.1%}")
    print(f"Total time: {total_dt:.1f}s, avg {total_dt/len(res_df):.1f}s/problem")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
