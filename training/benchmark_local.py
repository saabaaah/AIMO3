"""
Benchmark a model on reference.csv using vLLM SC-TIR locally.
Usage:
    python benchmark_local.py <model_path>

Example:
    python benchmark_local.py ./sft_final_model
    python benchmark_local.py ./final_model            # GRPO merged
    python benchmark_local.py Qwen/Qwen2.5-Math-7B-Instruct  # base
"""
import sys, os, re, signal, subprocess, tempfile, time
from collections import Counter
from contextlib import contextmanager

import pandas as pd
import torch
from vllm import LLM, SamplingParams

if len(sys.argv) < 2:
    print("Usage: python benchmark_local.py <model_path_or_hf_id>")
    sys.exit(1)

MODEL = sys.argv[1]
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", 10))  # test first N only by default
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 16))
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 2))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1280))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 4096))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
TOP_P = float(os.environ.get("TOP_P", 0.95))

print(f"=== Benchmark: {MODEL} ===")
print(f"N_PROBLEMS={N_PROBLEMS}, SAMPLES={NUM_SAMPLES}, ROUNDS={NUM_ROUNDS}, MAX_TOKENS={MAX_TOKENS}")

# ---- Safe Python REPL ----
BLOCKED = ["subprocess", "os.system", "__import__", "shutil", "open(", "eval(", "exec("]

@contextmanager
def _time_limit(seconds):
    def handler(*_): raise TimeoutError()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try: yield
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def execute_code(code, timeout=10):
    preamble = (
        "import math\nimport numpy as np\nimport sympy as sp\nfrom sympy import *\n"
        "from fractions import Fraction\nfrom itertools import permutations, combinations, product\n"
        "from functools import reduce\n"
    )
    full = preamble + code
    lines = full.strip().split("\n")
    last = lines[-1].split("#")[0].strip()
    if last and "print(" not in last and not last.startswith(("import","from","#","def","class","if","for","while","try","with")):
        lines[-1] = f"print({last})"; full = "\n".join(lines)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.py"); open(p,"w").write(full)
        try:
            with _time_limit(timeout):
                r = subprocess.run(["python3", p], capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0: return r.stdout.strip()[:1000]
            return r.stderr.strip()[-500:]
        except Exception as e:
            return str(e)[:200]

def extract_code_block(text):
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if not blocks: return ""
    code = blocks[-1].strip()
    for c in BLOCKED:
        if c in code: return ""
    return code

def extract_boxed(text):
    for prefix in [r"\boxed", r"\fbox"]:
        idx = text.rfind(prefix)
        if idx < 0: continue
        i, depth = idx, 0
        while i < len(text):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    start = text.index("{", idx) + 1
                    return text[start:i].strip()
            i += 1
    return ""

def to_int(text):
    if not text: return -1
    for rm in [r"\text{", "}", r"\mathrm{", "$", ",", " ", "%", "square", "units", "degrees"]:
        text = text.replace(rm, "")
    try:
        val = round(float(text))
        return max(0, min(99999, val))
    except Exception:
        return -1

# ---- Load model ----
print("\nLoading vLLM...")
llm = LLM(
    model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
    max_model_len=MAX_MODEL_LEN, trust_remote_code=True, gpu_memory_utilization=0.90,
)
tok = llm.get_tokenizer()

def build_prompt(problem):
    messages = [{"role": "user", "content": (
        "Solve this math problem step by step. "
        "Use Python code when calculations are needed.\n"
        "Put your final numerical answer in \\boxed{}.\n\n"
        f"Problem: {problem}"
    )}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def solve(problem):
    prompt = build_prompt(problem)
    cands = [prompt] * NUM_SAMPLES
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                        stop=["```output", "</s>", "<|im_end|>"], include_stop_str_in_output=True)
    for rnd in range(NUM_ROUNDS):
        outs = llm.generate(cands, sp, use_tqdm=False)
        cands = [o.prompt + o.outputs[0].text for o in outs]
        new_cands = []; any_code = False
        for text in cands:
            if text.rstrip().endswith("```output"):
                code = extract_code_block(text)
                if code:
                    result = execute_code(code)
                    text += f"\n{result}\n```\n"; any_code = True
                else:
                    text += "\n\n```\n"
            new_cands.append(text)
        cands = new_cands
        if not any_code: break
    answers = []
    for text in cands:
        b = extract_boxed(text)
        if b:
            v = to_int(b)
            if 0 <= v <= 99999: answers.append(v)
    if not answers: return 0, 0
    winner = Counter(answers).most_common(1)[0][0]
    return winner, len(answers)

# ---- Run ----
df = pd.read_csv("../data/reference.csv").head(N_PROBLEMS)
print(f"\nBenchmarking on {len(df)} problems...\n")

correct = 0; total_time = 0
for idx, row in df.iterrows():
    t0 = time.time()
    pred, n_extracted = solve(row["problem"])
    dt = time.time() - t0; total_time += dt
    true = int(row["answer"])
    ok = pred == true
    correct += ok
    status = "OK" if ok else "WRONG"
    print(f"#{idx+1:2d}: pred={pred:6d}, true={true:6d} [{n_extracted:2d}/{NUM_SAMPLES}] {dt:5.1f}s {status}")

print(f"\nScore: {correct}/{len(df)} = {correct/len(df):.0%}")
print(f"Total time: {total_time:.0f}s ({total_time/len(df):.0f}s/problem avg)")
