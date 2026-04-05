"""
Dump full model reasoning for one problem to understand failure modes.
Runs N samples sequentially with full output captured.
"""
import sys, os, re, signal, subprocess, tempfile
from contextlib import contextmanager
import pandas as pd
from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "./sft_final_model"
PROBLEM_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # which reference problem

# --- Load problem ---
df = pd.read_csv("../data/reference.csv")
row = df.iloc[PROBLEM_IDX]
problem = row["problem"]
true_answer = row["answer"]
print(f"=" * 70)
print(f"PROBLEM #{PROBLEM_IDX+1} (true answer: {true_answer})")
print(f"=" * 70)
print(problem)
print("=" * 70)

# --- Load model ---
print(f"\nLoading {MODEL}...\n")
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
          trust_remote_code=True, gpu_memory_utilization=0.90)
tok = llm.get_tokenizer()

# --- Build prompt ---
messages = [{"role": "user", "content": (
    "Solve this math problem step by step. "
    "Use Python code when calculations are needed.\n"
    "Put your final numerical answer in \\boxed{}.\n\n"
    f"Problem: {problem}"
)}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nPrompt tokens: {len(tok.encode(prompt))}")

# --- Generate with long max_tokens, no code execution (pure reasoning) ---
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=3000, n=3,
                    stop=["<|im_end|>", "</s>"])
out = llm.generate([prompt], sp, use_tqdm=False)

print(f"\n=== 3 sample reasoning chains ===\n")
for i, sample in enumerate(out[0].outputs):
    print(f"\n--- Sample {i+1} (finish={sample.finish_reason}, tokens={len(sample.token_ids)}) ---")
    print(sample.text)
    print(f"--- end sample {i+1} ---\n")
