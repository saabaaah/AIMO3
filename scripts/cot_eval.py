"""
Pure CoT evaluation with self-consistency — for reasoning models (R1-Distill, etc.)
that don't naturally use code blocks.
"""
import argparse
import sys
import time
from collections import Counter

import pandas as pd
import torch
from vllm import LLM, SamplingParams

sys.path.insert(0, "/home/ubuntu/AIMO3/scripts")
from sc_tir_eval import extract_boxed, to_int_answer


COT_SYSTEM = (
    "You are a math problem solver. Think carefully step by step. "
    "Put your final integer answer inside \\boxed{}."
)


def build_cot_prompt(tokenizer, problem: str, is_r1: bool) -> str:
    if is_r1:
        # R1-Distill chat template already prepends <think> — don't add system prompt
        messages = [{"role": "user", "content": problem + "\n\nPut your final integer answer inside \\boxed{}."}]
    else:
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": problem},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def solve_cot(llm, tokenizer, problem, n_samples, max_tokens, temp, top_p, is_r1):
    prompt = build_cot_prompt(tokenizer, problem, is_r1)
    sp = SamplingParams(
        temperature=temp, top_p=top_p, max_tokens=max_tokens, n=n_samples, seed=42,
    )
    outs = llm.generate([prompt], sp, use_tqdm=False)
    answers = []
    finish_counts = Counter()
    token_counts = []
    for o in outs[0].outputs:
        finish_counts[o.finish_reason] += 1
        token_counts.append(len(o.token_ids))
        boxed = extract_boxed(o.text)
        val = to_int_answer(boxed) if boxed else None
        if val is not None and 0 <= val <= 99999:
            answers.append(val)
    return answers, finish_counts, token_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--data", default="/home/ubuntu/AIMO3/data/val_amc.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    is_r1 = "r1" in args.model.lower() or "distill" in args.model.lower()

    print(f"Loading {args.model} (is_r1={is_r1}) on {torch.cuda.get_device_name(0)}...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Loaded in {time.time()-t0:.1f}s")

    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)

    results = []
    total_t0 = time.time()
    for i, row in df.iterrows():
        t0 = time.time()
        answers, finish_counts, tok_counts = solve_cot(
            llm, tokenizer, row["problem"], args.samples,
            args.max_tokens, args.temp, args.top_p, is_r1,
        )
        dt = time.time() - t0
        true = int(row["answer"])
        if answers:
            top = Counter(answers).most_common(1)[0]
            pred, vote = top
        else:
            pred, vote = 0, 0
        correct = pred == true
        avg_tok = sum(tok_counts) / len(tok_counts) if tok_counts else 0
        print(
            f"[{i+1}/{len(df)}] id={row['id']} true={true} pred={pred} "
            f"{'OK' if correct else 'XX'} | {dt:.1f}s | valid={len(answers)}/{args.samples} "
            f"vote={vote} avg_tok={avg_tok:.0f} finish={dict(finish_counts)} "
            f"top3={dict(Counter(answers).most_common(3))}",
            flush=True,
        )
        results.append({
            "id": row["id"], "true": true, "pred": pred, "correct": correct,
            "seconds": round(dt, 1), "valid_answers": len(answers),
            "vote": vote, "avg_tokens": round(avg_tok),
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
