#!/usr/bin/env python3
"""
Local test script - Test the solution on reference problems WITHOUT GPU.
Uses a simple chain-of-thought approach with sympy for calculations.
"""

import re
import sys
from pathlib import Path

# Add data directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

import pandas as pd
from sympy import *
from sympy.parsing.latex import parse_latex


def solve_with_sympy(problem: str) -> int:
    """
    Simple symbolic solver using sympy.
    This is a baseline - won't solve complex olympiad problems but useful for testing.
    """

    # Extract numbers from problem
    numbers = re.findall(r'\d+', problem)

    # Try to find answer pattern
    if "remainder when" in problem.lower() and "divided by" in problem.lower():
        # Modular arithmetic problem
        match = re.search(r'(\d+).*divided by.*?(\d+)', problem, re.IGNORECASE)
        if match:
            try:
                a, b = int(match.group(1)), int(match.group(2))
                return a % b
            except:
                pass

    # Look for simple arithmetic
    if numbers:
        # Return last number as guess (often the answer in simple problems)
        return int(numbers[-1]) % 100000

    return 0


def test_on_reference():
    """Test on reference problems."""

    ref_path = Path(__file__).parent.parent / "data" / "reference.csv"

    if not ref_path.exists():
        print(f"Reference file not found: {ref_path}")
        print("Run: kaggle competitions download -c ai-mathematical-olympiad-progress-prize-3")
        return

    df = pd.read_csv(ref_path)

    print("=" * 60)
    print("AIMO 3 - Local Test on Reference Problems")
    print("=" * 60)
    print(f"Problems: {len(df)}")
    print()

    correct = 0
    for idx, row in df.iterrows():
        problem = row["problem"][:100] + "..."  # Truncate for display
        true_answer = row["answer"]

        pred = solve_with_sympy(row["problem"])
        is_correct = pred == true_answer
        correct += is_correct

        status = "OK" if is_correct else "WRONG"
        print(f"#{idx+1} [{status}] True={true_answer}, Pred={pred}")

    print()
    print("=" * 60)
    print(f"Accuracy: {correct}/{len(df)} = {correct/len(df):.1%}")
    print("=" * 60)
    print()
    print("NOTE: This is a simple baseline. Real solution needs:")
    print("  1. LLM for reasoning (Qwen2.5-Math, DeepSeekMath)")
    print("  2. Code execution (Python REPL)")
    print("  3. Self-consistency (multiple samples + voting)")


if __name__ == "__main__":
    test_on_reference()
