"""
Generate self-edit trajectories using Groq API (llama3-70b-8192).
Takes SQuAD passages and GSM8K problems, produces synthetic SFT data.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict

from groq import Groq
from datasets import load_dataset
from tqdm import tqdm

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 5
RETRY_DELAY = 2.0  # seconds, doubles on each retry


def get_client():
    if not GROQ_API_KEY:
        raise ValueError("set GROQ_API_KEY env var")
    return Groq(api_key=GROQ_API_KEY)


def call_groq(client, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    delay = RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  FAILED after {MAX_RETRIES} attempts: {e}")
                return ""
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"  rate limited, waiting {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  error: {e}, retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2


# ─── SQuAD self-edit generation ───────────────────────────────

SQUAD_SYSTEM = (
    "You are a teaching assistant. Given a passage and a question, "
    "generate a concise training example that would help a student model "
    "learn to answer this type of question. Format your response as:\n"
    "PASSAGE: <key excerpt>\n"
    "QUESTION: <the question>\n"
    "REASONING: <step by step reasoning>\n"
    "ANSWER: <the answer>"
)

def generate_squad_edits(client, passages: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in tqdm(passages, desc="squad self-edits"):
            ctx = item["context"]
            q = item["question"]
            ans = item["answer"]

            prompt = f"Passage: {ctx}\n\nQuestion: {q}\n\nCorrect Answer: {ans}"
            response = call_groq(client, prompt, system=SQUAD_SYSTEM)

            if response:
                record = {
                    "source": "squad",
                    "context": ctx,
                    "question": q,
                    "gold_answer": ans,
                    "self_edit": response,
                    "instruction": f"Read the passage and answer: {q}\n\nPassage: {ctx}",
                    "response": response,
                }
                f.write(json.dumps(record) + "\n")


# ─── GSM8K self-edit generation ───────────────────────────────

MATH_SYSTEM = (
    "You are a math tutor. Given a math problem and its solution, "
    "generate a detailed chain-of-thought training example that would "
    "help a student model learn mathematical reasoning. Format:\n"
    "PROBLEM: <restate the problem>\n"
    "STEP 1: <first step>\n"
    "STEP 2: <second step>\n"
    "...\n"
    "ANSWER: <final numerical answer>"
)

def generate_math_edits(client, problems: List[Dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in tqdm(problems, desc="gsm8k self-edits"):
            q = item["instruction"]
            a = item["response"]

            prompt = f"Problem: {q}\n\nSolution: {a}"
            response = call_groq(client, prompt, system=MATH_SYSTEM)

            if response:
                record = {
                    "source": "gsm8k",
                    "question": q,
                    "gold_answer": a,
                    "self_edit": response,
                    "instruction": q,
                    "response": response,
                }
                f.write(json.dumps(record) + "\n")


# ─── Data loading helpers ─────────────────────────────────────

def load_squad_data(max_samples: int = 500) -> List[Dict]:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    samples = []
    for row in ds:
        if row["answers"]["text"]:
            samples.append({
                "context": row["context"],
                "question": row["question"],
                "answer": row["answers"]["text"][0],
            })
        if len(samples) >= max_samples:
            break
    print(f"loaded {len(samples)} squad samples")
    return samples


def load_gsm8k_data(max_samples: int = 500) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    samples = []
    for row in ds:
        samples.append({
            "instruction": row["question"],
            "response": row["answer"],
        })
        if len(samples) >= max_samples:
            break
    print(f"loaded {len(samples)} gsm8k samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="generate self-edit trajectories via groq")
    parser.add_argument("--output-dir", type=str, default="data/self_edits")
    parser.add_argument("--squad-samples", type=int, default=200)
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--dataset", choices=["squad", "gsm8k", "both"], default="both")
    args = parser.parse_args()

    out = Path(args.output_dir)
    client = get_client()

    if args.dataset in ("squad", "both"):
        squad_data = load_squad_data(args.squad_samples)
        generate_squad_edits(client, squad_data, out / "squad_edits.jsonl")

    if args.dataset in ("gsm8k", "both"):
        gsm8k_data = load_gsm8k_data(args.gsm8k_samples)
        generate_math_edits(client, gsm8k_data, out / "gsm8k_edits.jsonl")

    print("done.")


if __name__ == "__main__":
    main()
