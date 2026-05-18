# SLM Fine-Tuned with Knowledge Graphs

A research pipeline for **KG-grounded fine-tuning of small pre-trained language models**. The knowledge graph is not a retrieval sidecar — it is the structural substrate for training, evaluation, and alignment across the entire model-development lifecycle.

---

## Overview

This project investigates how structured knowledge can improve factual consistency, domain grounding, reasoning, sample efficiency, and hallucination resistance in compact language models. Knowledge graphs extracted from a user-supplied corpus are used across:

- corpus ingestion and chunking
- LLM-based KG extraction (user-defined prompts, configurable providers)
- KG validation and versioned snapshots
- pre-training, SFT, and DPO dataset generation
- KG-grounded reward functions for reinforcement learning
- structured evaluation and benchmarking

**Core implementation:** C++  
**Training and evaluation scripts:** Python (PyTorch / HuggingFace)

---

## Architecture

```
Input Documents
    │
    ▼
Corpus Ingestion & Chunking
    │
    ▼
External LLM KG Extraction          ← user-defined system + user prompts
    │                                  provider-configurable (OpenAI, Gemini)
    ▼                                  cached, versioned, replayable
Raw KG Candidates
    │
    ▼
KG Normalization & Validation
    │
    ▼
Versioned KG Snapshot
    ├────────────────────────────────┐
    ▼                                ▼
Dataset Generation           Evaluation Benchmark Generation
    │                                │
    ▼                                ▼
Pre-training / SFT / DPO      Benchmark Reports
    │
    ▼
Small Language Model
```

---

## Benchmark Results

Evaluated on `gpt2-small` fine-tuned with progressively richer KG grounding (token F1 on single-hop QA, 34 eval records):

| Baseline | Description | Avg F1 | Δ vs B0 |
|---|---|---:|---:|
| B0 | Zero-shot (frozen) | 0.1033 | — |
| B1 | Raw-text SFT | 0.0674 | −0.0359 |
| B2 | Graph-linearized pretrain | 0.0893 | −0.0140 |
| B3 | Graph-verbalized pretrain | 0.2022 | +0.0989 |
| B4 | KG-grounded SFT | **0.2184** | **+0.1151** |
| B5 | KG-grounded DPO | 0.1129 | +0.0096 |

Full reports are in [reports/](reports/).

---

## Repository Layout

```
configs/          YAML configuration files (corpus, KG extraction, model, training)
data/
  raw/            Input documents (text, Markdown, PDFs)
  processed/      Chunked and normalized corpus
  kg/             Extracted and validated KG snapshots
  datasets/       Generated pretrain / SFT / preference / eval datasets
documentation/    Full technical specification (slm-with-kgs.md)
prompts/          Versioned system and user prompt templates for KG extraction
reports/          Benchmark evaluation results (JSON + Markdown)
scripts/          Python training, evaluation, and utility scripts
src/              C++ pipeline source (corpus, extraction, KG, datasets, CLI)
tests/            C++ and Python tests with fixtures
```

Model checkpoints are excluded from this repository due to size. Use `scripts/download_checkpoint.py` to fetch them.

---

## Quick Start

### Prerequisites

- C++17 compiler, CMake ≥ 3.20
- Python ≥ 3.10
- An OpenAI or Gemini API key for KG extraction

### Build the C++ pipeline

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# 1. Ingest and chunk documents
./build/slmkg corpus ingest --config configs/corpus.yaml

# 2. Extract the knowledge graph
export OPENAI_API_KEY=<your-key>
./build/slmkg kg extract --config configs/kg_extraction.yaml

# 3. Validate and snapshot the KG
./build/slmkg kg validate --config configs/kg_schema.yaml

# 4. Generate training datasets
./build/slmkg dataset build --config configs/pretrain_dataset.yaml
./build/slmkg dataset build --config configs/sft_dataset.yaml

# 5. Fine-tune
python scripts/finetune_sft.py --config configs/sft.yaml
python scripts/finetune_dpo.py --config configs/dpo.yaml

# 6. Evaluate
python scripts/evaluate_checkpoint.py --config configs/eval_dataset.yaml

# 7. Compare baselines
python scripts/benchmark_compare.py
```

---

## Configuration

All pipeline behaviour is driven by YAML files in `configs/`. Key files:

| File | Purpose |
|---|---|
| `corpus.yaml` | Document ingestion, chunking, deduplication, splits |
| `kg_extraction.yaml` | Provider, model, prompts, caching, retry, concurrency |
| `kg_schema.yaml` | Entity types, relation types, arity constraints |
| `sft.yaml` / `dpo.yaml` | Fine-tuning hyperparameters |
| `eval_dataset.yaml` | Evaluation dataset and metrics |

API keys are read from environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`) and must never be committed.

---

## KG Extraction Prompts

Extraction prompts are fully user-defined. Place versioned prompt files in `prompts/`:

```
prompts/
  kg_system_prompt_v0.1.txt
  kg_user_prompt_v0.1.txt
```

The user prompt supports template variables (`{{chunk_text}}`, `{{kg_schema}}`, `{{allowed_entity_types}}`, etc.). Prompt hashes are stored with every extraction record for full reproducibility. Changing a prompt produces a new extraction manifest.

---

## KG Grounding

The KG is used as an explicit source of structure, constraints, and evaluation signals throughout:

| Stage | Role |
|---|---|
| Extraction | Candidate entities and relations with corpus provenance |
| Validation | Schema enforcement, normalization, rejected-candidate logging |
| Dataset generation | KG-derived pre-training, SFT, preference, and eval records |
| Fine-tuning | Graph-grounded instruction-response pairs |
| DPO | Graph-consistent chosen vs. graph-inconsistent rejected pairs |
| Evaluation | Graph-supported accuracy, contradiction rate, refusal quality |

Every generated example records the KG snapshot, entity IDs, and relation IDs used to create it.

---

## Testing

```bash
# C++ tests
cd build && ctest --output-on-failure

# Python tests
pytest tests/test_scripts.py -v
```

The test suite uses a mock extraction provider backed by fixture files in `tests/fixtures/mock_provider/`. No live API calls are required for tests.

---

## Python Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model training |
| `transformers` | Model and tokenizer loading |
| `datasets` | Dataset handling |
| `peft` | Parameter-efficient fine-tuning |
| `trl` | SFT and DPO trainers |
| `accelerate` | Distributed training support |
| `sentencepiece` | Tokenizer extension |

---

## Citation

If you use this pipeline in your research, please cite accordingly. The full technical specification is in [documentation/slm-with-kgs.md](documentation/slm-with-kgs.md).
