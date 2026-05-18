#!/usr/bin/env python3
"""Run inference on an eval JSONL and write a metrics report to JSON.

Called by `slmkg eval run` via std::system().

Usage:
  python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/b1_baseline_small \
    --eval-dataset data/datasets/eval/eval_v0.1.jsonl \
    --output reports/eval_b1.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(record: dict) -> str:
    parts = []
    if "instruction" in record:
        parts.append(record["instruction"])
    if "graph_context" in record and record["graph_context"]:
        parts.append("Graph context: " + " | ".join(record["graph_context"]))
    if "input" in record:
        parts.append("Question: " + record["input"])
    parts.append("Answer:")
    return "\n".join(parts)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and expected strings."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def run_evaluation(
    checkpoint: str,
    eval_dataset_path: str,
    output_path: str,
    max_new_tokens: int = 64,
    batch_size: int = 1,
    tokenizer_path: str = "",
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tokenizer_path or checkpoint
    print(f"[eval] Loading tokenizer from: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[eval] Loading model from: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
    model.eval()

    records = load_jsonl(eval_dataset_path)
    print(f"[eval] Loaded {len(records)} eval records from: {eval_dataset_path}")

    results = []
    t0 = time.time()

    for i, record in enumerate(records):
        prompt = build_prompt(record)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )

        generated = tok.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        expected = record.get("expected_output", "")
        f1 = token_f1(generated, expected)
        em = exact_match(generated, expected)

        results.append({
            "id": record.get("id", f"rec_{i}"),
            "task_type": record.get("task_type", ""),
            "input": record.get("input", ""),
            "expected_output": expected,
            "generated_output": generated,
            "token_f1": round(f1, 4),
            "exact_match": em,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            avg_f1 = sum(r["token_f1"] for r in results) / len(results)
            print(f"[eval] {i+1}/{len(records)} — running avg F1: {avg_f1:.4f}")

    elapsed = time.time() - t0

    avg_f1 = sum(r["token_f1"] for r in results) / len(results) if results else 0.0
    avg_em = sum(r["exact_match"] for r in results) / len(results) if results else 0.0

    task_f1: dict[str, list[float]] = {}
    for r in results:
        task_f1.setdefault(r["task_type"], []).append(r["token_f1"])
    per_task_f1 = {k: round(sum(v) / len(v), 4) for k, v in task_f1.items()}

    report = {
        "checkpoint": checkpoint,
        "eval_dataset": eval_dataset_path,
        "num_records": len(results),
        "avg_token_f1": round(avg_f1, 4),
        "avg_exact_match": round(avg_em, 4),
        "per_task_f1": per_task_f1,
        "elapsed_seconds": round(elapsed, 2),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[eval] avg token-F1: {avg_f1:.4f}  exact-match: {avg_em:.4f}")
    print(f"[eval] Report written to: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on eval JSONL.")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint directory to load")
    parser.add_argument("--eval-dataset", required=True,
                        help="Path to eval JSONL file")
    parser.add_argument("--output", required=True,
                        help="Path for the output JSON report")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tokenizer", default="",
                        help="Tokenizer directory (defaults to --checkpoint)")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        eval_dataset_path=args.eval_dataset,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
