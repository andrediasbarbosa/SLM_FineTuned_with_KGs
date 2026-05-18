#!/usr/bin/env python3
"""Compute KG-grounded reward components for each eval record.

Reward components (all in [-1, +1]):
  fact        — token-F1 of generated output vs expected_output
  consistency — fraction of output tokens present in graph_context (or 0 if absent)
  reasoning   — presence of multi-hop or relational reasoning markers
  uncertainty — correct refusal behaviour on missing-evidence records
  language    — normalised output length (penalises empty or excessively short)

Writes a JSON report with per-record reward breakdowns and aggregate stats.

Usage:
  python scripts/evaluate_kg_reward.py \
    --checkpoint checkpoints/b5_kg_dpo_small \
    --eval-dataset data/datasets/eval/eval_v0.1.jsonl \
    --output reports/reward_b5.json
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

REASONING_MARKERS = {
    "because", "therefore", "since", "which means", "implies",
    "leads to", "results in", "via", "through", "step",
}
REFUSAL_MARKERS = {
    "does not support", "no evidence", "cannot confirm",
    "not found", "insufficient", "no information",
    "not in the graph", "graph does not",
}


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def token_set(text: str) -> set[str]:
    return set(text.lower().split())


def reward_fact(generated: str, expected: str) -> float:
    """Token-F1 scaled to [-1, +1]."""
    pred = token_set(generated)
    gold = token_set(expected)
    if not pred or not gold:
        return -1.0
    common = pred & gold
    if not common:
        return -1.0
    p = len(common) / len(pred)
    r = len(common) / len(gold)
    f1 = 2 * p * r / (p + r)
    return round(2 * f1 - 1, 4)   # map [0,1] -> [-1,+1]


def reward_consistency(generated: str, graph_context: list[str]) -> float:
    """Fraction of non-trivial output tokens found in graph context, scaled [-1,+1]."""
    if not graph_context:
        return 0.0
    ctx_tokens = token_set(" ".join(graph_context))
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "of", "in", "to", "and", "or"}
    out_tokens = [t for t in generated.lower().split() if t not in STOPWORDS]
    if not out_tokens:
        return -1.0
    overlap = sum(1 for t in out_tokens if t in ctx_tokens)
    ratio = overlap / len(out_tokens)
    return round(2 * ratio - 1, 4)


def reward_reasoning(generated: str, task_type: str) -> float:
    """+1 if multi-hop task and reasoning markers present; 0 otherwise."""
    if task_type != "multi_hop_qa":
        return 0.0
    low = generated.lower()
    return 1.0 if any(m in low for m in REASONING_MARKERS) else -0.5


def reward_uncertainty(generated: str, task_type: str) -> float:
    """For missing-evidence tasks: +1 if refusal, -1 if hallucinated answer."""
    if task_type != "missing_evidence_refusal":
        return 0.0
    low = generated.lower()
    return 1.0 if any(m in low for m in REFUSAL_MARKERS) else -1.0


def reward_language(generated: str, max_new_tokens: int = 64) -> float:
    """Length score: penalise very short outputs; plateau at half of max_new_tokens."""
    n = len(generated.split())
    if n == 0:
        return -1.0
    target = max(1, max_new_tokens // 2)
    ratio = min(n / target, 1.0)
    return round(2 * ratio - 1, 4)


COMPONENT_WEIGHTS = {
    "fact":        0.40,
    "consistency": 0.25,
    "reasoning":   0.15,
    "uncertainty": 0.10,
    "language":    0.10,
}


def compute_reward(components: dict[str, float]) -> float:
    total = sum(COMPONENT_WEIGHTS[k] * v for k, v in components.items())
    return round(total, 4)


def run_reward_evaluation(
    checkpoint: str,
    eval_dataset_path: str,
    output_path: str,
    max_new_tokens: int = 64,
    tokenizer_path: str = "",
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tokenizer_path or checkpoint
    print(f"[reward] Loading tokenizer from: {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[reward] Loading model from: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
    model.eval()

    records = load_jsonl(eval_dataset_path)
    print(f"[reward] {len(records)} eval records from: {eval_dataset_path}")

    results = []
    t0 = time.time()

    for i, record in enumerate(records):
        parts = []
        if "instruction" in record:
            parts.append(record["instruction"])
        if "graph_context" in record and record["graph_context"]:
            parts.append("Graph context: " + " | ".join(record["graph_context"]))
        if "input" in record:
            parts.append("Question: " + record["input"])
        parts.append("Answer:")
        prompt = "\n".join(parts)

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

        expected   = record.get("expected_output", "")
        task_type  = record.get("task_type", "")
        graph_ctx  = record.get("graph_context", [])

        components = {
            "fact":        reward_fact(generated, expected),
            "consistency": reward_consistency(generated, graph_ctx),
            "reasoning":   reward_reasoning(generated, task_type),
            "uncertainty": reward_uncertainty(generated, task_type),
            "language":    reward_language(generated, max_new_tokens),
        }
        total = compute_reward(components)

        results.append({
            "id":                   record.get("id", f"rec_{i}"),
            "task_type":            task_type,
            "prompt_id":            record.get("id", ""),
            "candidate":            generated,
            "expected_output":      expected,
            "reward_total":         total,
            "reward_components":    components,
            "reward_function_version": "kg_reward_v0.1",
            "kg_snapshot":          record.get("kg_snapshot", ""),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            avg = sum(r["reward_total"] for r in results) / len(results)
            print(f"[reward] {i+1}/{len(records)} — running avg reward: {avg:.4f}")

    elapsed = time.time() - t0

    avg_reward = sum(r["reward_total"] for r in results) / len(results) if results else 0.0
    avg_components: dict[str, float] = {}
    for comp in COMPONENT_WEIGHTS:
        vals = [r["reward_components"][comp] for r in results]
        avg_components[comp] = round(sum(vals) / len(vals), 4) if vals else 0.0

    report = {
        "checkpoint":           checkpoint,
        "eval_dataset":         eval_dataset_path,
        "num_records":          len(results),
        "avg_reward_total":     round(avg_reward, 4),
        "avg_reward_components": avg_components,
        "component_weights":    COMPONENT_WEIGHTS,
        "elapsed_seconds":      round(elapsed, 2),
        "evaluated_at":         datetime.now(timezone.utc).isoformat(),
        "results":              results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[reward] avg total reward: {avg_reward:.4f}")
    for comp, val in avg_components.items():
        print(f"  {comp:12s}: {val:.4f}  (weight {COMPONENT_WEIGHTS[comp]:.2f})")
    print(f"[reward] Report written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KG reward evaluation.")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--eval-dataset",  required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--tokenizer",     default="")
    args = parser.parse_args()

    run_reward_evaluation(
        checkpoint=args.checkpoint,
        eval_dataset_path=args.eval_dataset,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
