# Benchmark Comparison

_Generated: 2026-05-17 14:37 UTC_

## Summary

| Baseline | Checkpoint | Records | Avg F1 | Exact Match |
|---|---|---|---|---|
| B0 — zero-shot (frozen) | `gpt2-small` | 34 | **0.1033** | 0.0000 |
| B1 — raw-text SFT | `b1_baseline_small` | 34 | **0.0674** | 0.0000 |
| B2 — graph-linearized pretrain | `b2_linearized_small` | 34 | **0.0893** | 0.0000 |
| B3 — graph-verbalized pretrain | `b3_verbalized_small` | 34 | **0.2022** | 0.0000 |
| B4 — KG-grounded SFT | `b4_kg_sft_small` | 34 | **0.2184** | 0.0000 |
| B5 — KG DPO | `b5_kg_dpo_small` | 34 | **0.1129** | 0.0000 |

## F1 Delta vs B0

| Baseline | Avg F1 | Δ F1 vs B0 |
|---|---|---|
| B0 — zero-shot (frozen) | 0.1033 | +0.0000 |
| B1 — raw-text SFT | 0.0674 | -0.0359 |
| B2 — graph-linearized pretrain | 0.0893 | -0.0140 |
| B3 — graph-verbalized pretrain | 0.2022 | +0.0989 |
| B4 — KG-grounded SFT | 0.2184 | +0.1151 |
| B5 — KG DPO | 0.1129 | +0.0096 |

## Per-Task Token F1

| Baseline | `single_hop_qa` |
|---|---|
| B0 — zero-shot (frozen) | 0.1033 |
| B1 — raw-text SFT | 0.0674 |
| B2 — graph-linearized pretrain | 0.0893 |
| B3 — graph-verbalized pretrain | 0.2022 |
| B4 — KG-grounded SFT | 0.2184 |
| B5 — KG DPO | 0.1129 |

## Run Details

### B0 — zero-shot (frozen)
- Report: `reports/eval_b0.json`
- Checkpoint: `checkpoints/base/gpt2-small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.1033
- Exact match: 0.0000
- Elapsed: 202.8s
- Evaluated at: 2026-05-16T12:16:03.591835+00:00

### B1 — raw-text SFT
- Report: `reports/eval_b1.json`
- Checkpoint: `checkpoints/b1_baseline_small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.0674
- Exact match: 0.0000
- Elapsed: 295.18s
- Evaluated at: 2026-05-16T16:42:09.973036+00:00

### B2 — graph-linearized pretrain
- Report: `reports/eval_b2_linearized.json`
- Checkpoint: `checkpoints/b2_linearized_small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.0893
- Exact match: 0.0000
- Elapsed: 327.69s
- Evaluated at: 2026-05-16T18:14:32.862068+00:00

### B3 — graph-verbalized pretrain
- Report: `reports/eval_b3_verbalized.json`
- Checkpoint: `checkpoints/b3_verbalized_small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.2022
- Exact match: 0.0000
- Elapsed: 97.28s
- Evaluated at: 2026-05-17T13:21:39.405004+00:00

### B4 — KG-grounded SFT
- Report: `reports/eval_b4_kg_sft.json`
- Checkpoint: `checkpoints/b4_kg_sft_small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.2184
- Exact match: 0.0000
- Elapsed: 67.02s
- Evaluated at: 2026-05-16T20:14:46.148970+00:00

### B5 — KG DPO
- Report: `reports/eval_b5_kg_dpo.json`
- Checkpoint: `checkpoints/b5_kg_dpo_small`
- Eval dataset: `data/datasets/eval/eval_v0.1.jsonl`
- Records: 34
- Avg token-F1: 0.1129
- Exact match: 0.0000
- Elapsed: 319.27s
- Evaluated at: 2026-05-17T13:57:31.518364+00:00
