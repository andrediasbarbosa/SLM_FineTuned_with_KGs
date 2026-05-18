#!/usr/bin/env python3
"""Compare multiple eval JSON reports and write a Markdown summary table.

Usage:
  python scripts/benchmark_compare.py \
    --runs reports/eval_b0.json reports/eval_b1.json reports/eval_b4_kg_sft.json \
    --output reports/benchmark_comparison.md
"""

import argparse
import json
import os
from datetime import datetime, timezone


BASELINE_LABELS = {
    "eval_b0":           "B0 — zero-shot (frozen)",
    "eval_b1":           "B1 — raw-text SFT",
    "eval_b2_linearized": "B2 — graph-linearized pretrain",
    "eval_b3_verbalized": "B3 — graph-verbalized pretrain",
    "eval_b4_kg_sft":    "B4 — KG-grounded SFT",
    "eval_b5_kg_dpo":    "B5 — KG DPO",
}


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def label_for(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return BASELINE_LABELS.get(stem, stem)


def all_task_types(reports: list[dict]) -> list[str]:
    tasks = set()
    for r in reports:
        tasks.update(r.get("per_task_f1", {}).keys())
    return sorted(tasks)


def render_markdown(paths: list[str], reports: list[dict]) -> str:
    labels = [label_for(p) for p in paths]
    tasks = all_task_types(reports)

    lines = []
    lines.append("# Benchmark Comparison\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # Summary table
    lines.append("## Summary\n")
    header = "| Baseline | Checkpoint | Records | Avg F1 | Exact Match |"
    sep    = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for label, path, r in zip(labels, paths, reports):
        ckpt = os.path.basename(r.get("checkpoint", "—"))
        lines.append(
            f"| {label} | `{ckpt}` | {r['num_records']} "
            f"| **{r['avg_token_f1']:.4f}** | {r['avg_exact_match']:.4f} |"
        )
    lines.append("")

    # Delta table (relative to B0)
    if len(reports) > 1:
        b0_f1 = reports[0]["avg_token_f1"]
        lines.append("## F1 Delta vs B0\n")
        header2 = "| Baseline | Avg F1 | Δ F1 vs B0 |"
        sep2    = "|---|---|---|"
        lines.append(header2)
        lines.append(sep2)
        for label, r in zip(labels, reports):
            delta = r["avg_token_f1"] - b0_f1
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {label} | {r['avg_token_f1']:.4f} | {sign}{delta:.4f} |")
        lines.append("")

    # Per-task breakdown
    if tasks:
        lines.append("## Per-Task Token F1\n")
        col_headers = " | ".join(f"`{t}`" for t in tasks)
        header3 = f"| Baseline | {col_headers} |"
        sep3    = "|---|" + "---|" * len(tasks)
        lines.append(header3)
        lines.append(sep3)
        for label, r in zip(labels, reports):
            per = r.get("per_task_f1", {})
            cols = " | ".join(f"{per.get(t, 0):.4f}" for t in tasks)
            lines.append(f"| {label} | {cols} |")
        lines.append("")

    # Individual report details
    lines.append("## Run Details\n")
    for label, path, r in zip(labels, paths, reports):
        lines.append(f"### {label}")
        lines.append(f"- Report: `{path}`")
        lines.append(f"- Checkpoint: `{r.get('checkpoint', '—')}`")
        lines.append(f"- Eval dataset: `{r.get('eval_dataset', '—')}`")
        lines.append(f"- Records: {r['num_records']}")
        lines.append(f"- Avg token-F1: {r['avg_token_f1']:.4f}")
        lines.append(f"- Exact match: {r['avg_exact_match']:.4f}")
        lines.append(f"- Elapsed: {r.get('elapsed_seconds', '—')}s")
        lines.append(f"- Evaluated at: {r.get('evaluated_at', '—')}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval JSON reports.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to eval JSON reports (in baseline order)")
    parser.add_argument("--output", required=True,
                        help="Output Markdown file path")
    args = parser.parse_args()

    reports = []
    for path in args.runs:
        if not os.path.exists(path):
            print(f"[compare] WARNING: report not found, skipping: {path}")
            continue
        reports.append(load_report(path))

    if not reports:
        print("[compare] No valid reports found.")
        return

    valid_paths = [p for p in args.runs if os.path.exists(p)]
    md = render_markdown(valid_paths, reports)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)

    print(f"[compare] {len(reports)} reports compared -> {args.output}")

    # Also print summary to stdout
    print()
    for path, r in zip(valid_paths, reports):
        print(f"  {label_for(path):35s}  F1={r['avg_token_f1']:.4f}  EM={r['avg_exact_match']:.4f}")


if __name__ == "__main__":
    main()
