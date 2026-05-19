# SLM + Knowledge Graphs — Pipeline Documentation

This document explains how the pipeline works end-to-end and provides step-by-step instructions for fine-tuning a base model on a new corpus.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Stages](#2-pipeline-stages)
   - [Stage 1 — Corpus Ingestion and Chunking](#stage-1--corpus-ingestion-and-chunking)
   - [Stage 2 — KG Extraction](#stage-2--kg-extraction)
   - [Stage 3 — KG Normalization and Validation](#stage-3--kg-normalization-and-validation)
   - [Stage 4 — Dataset Generation](#stage-4--dataset-generation)
   - [Stage 5 — Tokenizer Extension (optional)](#stage-5--tokenizer-extension-optional)
   - [Stage 6 — Fine-tuning (SFT and DPO)](#stage-6--fine-tuning-sft-and-dpo)
   - [Stage 7 — Evaluation](#stage-7--evaluation)
   - [Stage 8 — Benchmark Comparison](#stage-8--benchmark-comparison)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Repository Layout](#4-repository-layout)
5. [Prerequisites](#5-prerequisites)
6. [How to Fine-tune on a New Corpus](#6-how-to-fine-tune-on-a-new-corpus)
   - [Step 1 — Prepare your documents](#step-1--prepare-your-documents)
   - [Step 2 — Configure corpus ingestion](#step-2--configure-corpus-ingestion)
   - [Step 3 — Configure the KG schema](#step-3--configure-the-kg-schema)
   - [Step 4 — Configure KG extraction](#step-4--configure-kg-extraction)
   - [Step 5 — Write or update extraction prompts](#step-5--write-or-update-extraction-prompts)
   - [Step 6 — Run the C++ pipeline](#step-6--run-the-c-pipeline)
   - [Step 7 — Configure training datasets](#step-7--configure-training-datasets)
   - [Step 8 — Configure model and training hyperparameters](#step-8--configure-model-and-training-hyperparameters)
   - [Step 9 — Fine-tune](#step-9--fine-tune)
   - [Step 10 — Evaluate and compare](#step-10--evaluate-and-compare)
7. [Configuration Reference](#7-configuration-reference)
8. [KG Grounding Across the Pipeline](#8-kg-grounding-across-the-pipeline)
9. [Benchmark Results](#9-benchmark-results)
10. [Testing](#10-testing)

---

## 1. Overview

This project is a research pipeline for **KG-grounded fine-tuning of small pre-trained language models**. The knowledge graph (KG) is not an inference-time retrieval sidecar — it is the structural substrate for every stage of model development: corpus ingestion, dataset generation, training, alignment, and evaluation.

The research question is: does anchoring every training example, preference label, and evaluation signal to validated graph facts improve factual consistency, domain grounding, reasoning, and hallucination resistance in compact models?

**Core pipeline:** C++ (corpus, extraction, KG store, dataset builders, CLI)  
**Training and evaluation:** Python (PyTorch / HuggingFace / trl / peft)

---

## 2. Pipeline Stages

### Stage 1 — Corpus Ingestion and Chunking

**Binary:** `slmkg corpus ingest`  
**Config:** `configs/corpus.yaml`

The pipeline starts with raw documents (`data/raw/`). Supported formats are `.txt`, `.md`, `.jsonl`, and `.pdf`. Each document is read, cleaned, and split into overlapping text chunks.

Key behaviours:

- **Chunking:** configurable minimum/maximum chunk size (in characters) and overlap. Default: 300–2000 chars with 100-char overlap.
- **Deduplication:** exact SHA-256 hash deduplication at the chunk level, so duplicate passages extracted from multiple documents do not pollute the KG.
- **Splits:** documents are assigned to train/validation/test splits by document ID (not by chunk), so no document leaks across splits. Default: 80/10/10.
- **Quality checks:** language score threshold, malformed-JSONL detection, unreadable-file detection, duplicate-ID detection.
- **Manifest:** a `corpus_manifest.json` is written to the output directory recording the document list, chunk counts, split assignments, and SHA-256 hashes of every source file.

Output: `data/processed/<corpus_id>/chunks.jsonl` — one JSON object per chunk with fields `document_id`, `chunk_id`, `text`, `split`, and `char_hash`.

---

### Stage 2 — KG Extraction

**Binary:** `slmkg kg extract`  
**Config:** `configs/kg_extraction.yaml`

Each chunk is sent to an external LLM (OpenAI or Gemini) with a user-defined system prompt and user prompt. The LLM returns a JSON object containing entities and relations found in the chunk.

Key behaviours:

- **Provider abstraction:** `openai`, `gemini`, `mock` (fixture-backed, for testing), or `replay` (re-use a previous API response).
- **Prompt versioning:** system and user prompts are loaded from files in `prompts/`. Every extraction record stores the SHA-256 hash of the prompt files used, so a prompt change automatically invalidates the cache.
- **Response caching:** responses are stored on disk under `.cache/kg_extraction/` keyed by a hash of `(chunk_hash, prompt_hash, provider, model)`. Re-running extraction with the same inputs is free.
- **Retry and backoff:** configurable max attempts and exponential backoff.
- **Concurrency:** up to N parallel in-flight API requests (default: 4).

The user prompt template supports these variables:

| Variable | Description |
|---|---|
| `{{document_id}}` | Source document identifier |
| `{{chunk_id}}` | Chunk identifier within the document |
| `{{chunk_text}}` | The text of the chunk |
| `{{kg_schema}}` | Rendered schema of allowed entity and relation types |
| `{{allowed_entity_types}}` | Comma-separated entity type list |

Output: raw extraction candidates stored per-chunk, indexed by an extraction manifest.

---

### Stage 3 — KG Normalization and Validation

**Binary:** `slmkg kg validate`  
**Config:** `configs/kg_schema.yaml`

Raw extraction candidates are normalized and validated against the KG schema before any training data is generated.

Steps:

1. **Entity normalization:** canonical names are lowercased and whitespace-collapsed; aliases are resolved to a single canonical entity ID.
2. **Schema validation:** entity types and relation types must appear in the schema. Out-of-schema candidates are rejected and logged.
3. **Arity checks:** relation types have configurable minimum and maximum arities for head and tail.
4. **Provenance recording:** every accepted entity and relation records the source document ID, chunk ID, extraction prompt hash, and LLM provider response.
5. **Rejected-candidate log:** rejections are stored separately for audit and ablation analysis.
6. **Versioned snapshot:** the validated KG is serialized to `data/kg/<snapshot_id>/` including a `graph_snapshot.json` summary and per-entity / per-relation records.

The validated KG snapshot is the **single authoritative source** for all downstream dataset generation and evaluation. Raw extraction output is never used directly for training.

---

### Stage 4 — Dataset Generation

**Binary:** `slmkg dataset build`  
**Configs:** `configs/pretrain_dataset.yaml`, `configs/sft_dataset.yaml`, `configs/eval_dataset.yaml`

Four dataset types are generated from the validated KG snapshot. Each generated record stores the KG snapshot ID, entity IDs, and relation IDs used to create it, enabling later traceability analysis.

**Pre-training dataset** (`pretrain_builder`)  
Graph-linearized or graph-verbalized text sequences for continued pre-training. Two linearization modes:

- *Linearized:* structured triples serialized as `[ENTITY] head [RELATION] relation [ENTITY] tail [GRAPH_END]`
- *Verbalized:* natural-language sentences generated from triples (e.g. "GPT-2 is a Model that uses the WebText Dataset.")

Multi-hop paths (chains of two or more relations through the graph) are also included up to the configured `max_hop_paths` limit.

**SFT dataset** (`sft_builder`)  
Instruction-response pairs grounded in graph facts. Each record has:
- `instruction` — a domain instruction (e.g. "Answer based on your knowledge of the domain.")
- `graph_context` — a list of verbalized graph facts relevant to the question
- `input` — a single-hop or multi-hop question generated from the KG
- `output` — the KG-supported answer
- `split` — train / validation / test

A configurable `refusal_ratio` (default 0.15) adds examples where the KG does not contain enough evidence, training the model to express appropriate uncertainty.

**Preference dataset** (`preference_builder`)  
DPO-format records with `prompt`, `chosen` (graph-consistent response), and `rejected` (graph-inconsistent or corrupted response). Rejection is generated by substituting incorrect entities or relations drawn from the KG's negative-sampling pool.

**Eval dataset** (`eval_builder`)  
Single-hop QA records held out from training, with `expected_output` drawn directly from validated KG facts. Used by the evaluation script.

---

### Stage 5 — Tokenizer Extension (optional)

**Script:** `scripts/extend_tokenizer.py`

If you want the model to represent KG structure explicitly in token space, this script adds six special tokens to the base tokenizer:

```
[ENTITY]  [RELATION]  [GRAPH_START]  [GRAPH_END]  [HOP]  [EVIDENCE]
```

This is recommended when using the linearized pre-training dataset. The extended tokenizer is saved to `checkpoints/tokenizer_kg_v0.1/` and referenced via `tokenizer_path` in `configs/model.yaml`.

```bash
python scripts/extend_tokenizer.py \
  --base-checkpoint checkpoints/base/gpt2-small \
  --output checkpoints/tokenizer_kg_v0.1 \
  --model-checkpoint checkpoints/base/gpt2-small   # optional: also resize embeddings
```

---

### Stage 6 — Fine-tuning (SFT and DPO)

**Scripts:** `scripts/finetune_sft.py`, `scripts/finetune_dpo.py`  
**Configs:** `configs/model.yaml`, `configs/sft.yaml`, `configs/dpo.yaml`

Both scripts use LoRA (via `peft`) to fine-tune only a small fraction of model parameters. The LoRA adapter is saved alongside a `training_manifest.json` recording hyperparameters, dataset path, KG snapshot, trainable parameter count, final loss, and wall-clock time.

**SFT fine-tuning**

```bash
python scripts/finetune_sft.py \
  --base-checkpoint checkpoints/base/gpt2-small \
  --dataset data/datasets/sft/sft_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/sft.yaml \
  --output checkpoints/my_kg_sft
```

The script filters the dataset to the `split: train` records by default. SFT records with a `text` field are used directly; records with `instruction` / `graph_context` / `input` / `output` fields are formatted as:

```
<instruction>
Graph context: <fact1> | <fact2> | ...
Question: <input>
Answer: <output>
```

**DPO fine-tuning**

DPO takes a KG-SFT checkpoint as its starting point and trains on the preference dataset:

```bash
python scripts/finetune_dpo.py \
  --base-checkpoint checkpoints/my_kg_sft \
  --preference-dataset data/datasets/preference/preference_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/dpo.yaml \
  --output checkpoints/my_kg_dpo
```

---

### Stage 7 — Evaluation

**Script:** `scripts/evaluate_checkpoint.py`

Runs greedy decoding on every record in the eval JSONL and computes token-level F1 and exact match against the expected KG-grounded answer.

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/my_kg_sft \
  --eval-dataset data/datasets/eval/eval_v0.1.jsonl \
  --output reports/eval_my_kg_sft.json
```

The output JSON records per-record predictions and aggregate metrics broken down by task type.

---

### Stage 8 — Benchmark Comparison

**Script:** `scripts/benchmark_compare.py`

Compares multiple eval JSON reports and renders a Markdown table with F1 deltas relative to the B0 zero-shot baseline.

```bash
python scripts/benchmark_compare.py \
  --runs reports/eval_b0.json reports/eval_my_kg_sft.json \
  --output reports/my_comparison.md
```

---

## 3. Architecture Diagram

```
data/raw/
  (txt, md, pdf, jsonl)
        │
        ▼
  slmkg corpus ingest
  (chunking, dedup, splits)
        │
        ▼
  data/processed/<corpus_id>/chunks.jsonl
        │
        ▼
  slmkg kg extract
  (external LLM, versioned prompts, cached)
        │
        ▼
  raw extraction candidates
        │
        ▼
  slmkg kg validate
  (normalization, schema checks, provenance)
        │
        ▼
  data/kg/<snapshot_id>/   ←── single authoritative artifact
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  slmkg dataset build              slmkg dataset build
  (pretrain / SFT / preference)    (eval)
        │                                  │
        ▼                                  ▼
  data/datasets/                   data/datasets/eval/
        │
        ├── (optional) extend_tokenizer.py
        │
        ├── finetune_sft.py
        │       │
        │       ▼
        │   checkpoints/<sft>/
        │
        └── finetune_dpo.py
                │
                ▼
           checkpoints/<dpo>/
                │
                ▼
         evaluate_checkpoint.py
                │
                ▼
         reports/eval_*.json
                │
                ▼
         benchmark_compare.py
                │
                ▼
         reports/benchmark_comparison.md
```

---

## 4. Repository Layout

```
configs/           YAML configuration files (see §7)
data/
  raw/             Input documents (.txt, .md, .pdf, .jsonl)
  processed/       Chunked corpus output
  kg/              Validated KG snapshots
  datasets/        Generated pretrain / SFT / preference / eval datasets
documentation/     Technical specification and this file
prompts/           Versioned system and user prompt templates
reports/           Benchmark evaluation results (JSON + Markdown)
scripts/           Python training, evaluation, and utility scripts
src/               C++ pipeline source (corpus, extraction, KG, datasets, CLI)
tests/             C++ and Python tests with fixtures
checkpoints/       LoRA adapters and tokenizer (not committed; use download_checkpoint.py)
```

---

## 5. Prerequisites

**C++ pipeline**
- C++17 compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 2019+)
- CMake ≥ 3.20

**Python training and evaluation**
- Python ≥ 3.10
- Packages: `torch`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`, `sentencepiece`, `pyyaml`

**API key** — one of the following, depending on `kg_extraction.yaml`:
- `OPENAI_API_KEY` for the OpenAI provider
- `GEMINI_API_KEY` for the Gemini provider

**Build the C++ pipeline**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Set up the Python environment**

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 6. How to Fine-tune on a New Corpus

This section walks through replacing the existing corpus with your own documents and producing a new fine-tuned model from scratch.

### Step 1 — Prepare your documents

Place your raw documents under `data/raw/`. Supported formats:

| Format | Notes |
|---|---|
| `.txt` | Plain UTF-8 text |
| `.md` | Markdown (rendered as plain text during chunking) |
| `.pdf` | PDFs are parsed page by page |
| `.jsonl` | One JSON object per line; must include a `text` field |

There is no required directory structure inside `data/raw/`. Documents are discovered recursively.

### Step 2 — Configure corpus ingestion

Edit `configs/corpus.yaml` (or copy it and give it a new `corpus.id`):

```yaml
corpus:
  id: my_corpus_v1                    # change this for a new corpus version
  source_root: data/raw               # where your documents live
  output_root: data/processed/my_corpus_v1

  chunking:
    min_chunk_chars: 300              # increase for longer documents
    max_chunk_chars: 2000
    overlap_chars: 100

  splits:
    train: 0.80
    validation: 0.10
    test: 0.10
    seed: 42
    split_by: document_id             # keeps whole documents in one split
```

A larger `max_chunk_chars` gives the LLM more context per extraction call but costs more tokens. A good starting range for scientific papers is 800–1500 chars.

### Step 3 — Configure the KG schema

Edit `configs/kg_schema.yaml` to match your domain. The schema lists:

- **Entity types** — the categories of things you want to extract (e.g. `Concept`, `Method`, `Person`). Keep only the types that are meaningful for your corpus.
- **Relation types** — the links between entities, with `domain` and `range` constraints and arity bounds.

You do not need to enumerate every possible entity — the default schema covers most academic/technical domains. Remove types that have no analogue in your corpus to reduce noise.

If you change the schema, increment `schema_version` to avoid reusing cached extractions from the old schema.

### Step 4 — Configure KG extraction

Edit `configs/kg_extraction.yaml`:

```yaml
kg_extraction:
  provider: openai               # openai | gemini
  cache_enabled: true
  cache_dir: .cache/kg_extraction

  prompts:
    prompt_set_id: my_prompts_v1
    system_prompt_path: prompts/kg_system_prompt_v0.1.txt
    user_prompt_path: prompts/kg_user_prompt_v0.2.txt

  output:
    schema_path: configs/kg_schema.yaml
    strict_validation: false     # set true to reject out-of-schema types hard

  concurrency:
    max_concurrent_requests: 4   # reduce if hitting rate limits
```

Set your API key in the environment — never put it in the config file:

```bash
export OPENAI_API_KEY=sk-...
# or
export GEMINI_API_KEY=...
```

### Step 5 — Write or update extraction prompts

The system prompt (`prompts/kg_system_prompt_v0.1.txt`) sets the LLM's role. The default instructs it to extract only corpus-grounded facts and return valid JSON. You can keep this as-is.

The user prompt (`prompts/kg_user_prompt_v0.2.txt`) controls what the LLM extracts. Copy it to a new version if you want to iterate without invalidating existing cache entries:

```bash
cp prompts/kg_user_prompt_v0.2.txt prompts/kg_user_prompt_v1.0.txt
```

Then update `user_prompt_path` in `kg_extraction.yaml` to point at the new file.

The template variables available in the user prompt are:

```
{{document_id}}           source document identifier
{{chunk_id}}              chunk identifier
{{chunk_text}}            the text to extract from
{{kg_schema}}             rendered schema (entity + relation types)
{{allowed_entity_types}}  comma-separated entity type names
```

Prompt changes are hashed automatically. Any extraction run that sees a new prompt hash produces a fresh manifest and bypasses the cache.

### Step 6 — Run the C++ pipeline

Run the four C++ pipeline commands in order:

```bash
# 1. Ingest and chunk documents
./build/slmkg corpus ingest --config configs/corpus.yaml

# 2. Extract the knowledge graph  (requires API key in environment)
./build/slmkg kg extract --config configs/kg_extraction.yaml

# 3. Validate and snapshot the KG
./build/slmkg kg validate --config configs/kg_schema.yaml

# 4. Generate training and evaluation datasets
./build/slmkg dataset build --config configs/pretrain_dataset.yaml
./build/slmkg dataset build --config configs/sft_dataset.yaml
./build/slmkg dataset build --config configs/eval_dataset.yaml
```

After step 2, check `.cache/kg_extraction/` to confirm that responses are being cached. Subsequent runs with the same chunks and prompts will not make API calls.

After step 3, inspect `data/kg/<snapshot_id>/` to verify that entities and relations were extracted. If the snapshot is empty, review the user prompt and `strict_validation` setting.

### Step 7 — Configure training datasets

Edit `configs/pretrain_dataset.yaml` and `configs/sft_dataset.yaml`. The key field to update is `kg_snapshot_id` — it must match the snapshot ID produced by step 6:

```yaml
dataset:
  kg_snapshot_id: my_corpus_kg_v1    # must match data/kg/<this-id>/
  corpus_chunks_path: data/processed/my_corpus_v1/chunks.jsonl
  max_records: 20000
```

Reduce `max_records` if your corpus is small; the builders will simply generate fewer examples.

### Step 8 — Configure model and training hyperparameters

Edit `configs/model.yaml` to specify the base model and LoRA settings:

```yaml
# Leave tokenizer_path empty to use the base checkpoint's tokenizer,
# or point it at a custom tokenizer saved by extend_tokenizer.py.
tokenizer_path: ""

lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: ["c_attn"]    # GPT-2 attention projection;
                                 # change for other architectures (e.g. ["q_proj","v_proj"] for LLaMA)
```

Edit `configs/sft.yaml` to tune the SFT hyperparameters:

```yaml
num_epochs: 3
batch_size: 4
learning_rate: 2.0e-4
max_seq_length: 512
warmup_steps: 50
```

For a domain that differs significantly from the base model's pre-training data, consider increasing `num_epochs` to 5 and reducing `learning_rate` slightly (e.g. `1.0e-4`).

### Step 9 — Fine-tune

**Optional: extend the tokenizer** (recommended if using linearized pre-training)

```bash
python scripts/extend_tokenizer.py \
  --base-checkpoint checkpoints/base/gpt2-small \
  --output checkpoints/tokenizer_kg_v1 \
  --model-checkpoint checkpoints/base/gpt2-small

# Then set in configs/model.yaml:
#   tokenizer_path: checkpoints/tokenizer_kg_v1
```

**SFT on the KG-grounded dataset**

```bash
python scripts/finetune_sft.py \
  --base-checkpoint checkpoints/base/gpt2-small \
  --dataset data/datasets/sft/sft_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/sft.yaml \
  --output checkpoints/my_kg_sft
```

**DPO alignment (optional, builds on the SFT checkpoint)**

```bash
python scripts/finetune_dpo.py \
  --base-checkpoint checkpoints/my_kg_sft \
  --preference-dataset data/datasets/preference/preference_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/dpo.yaml \
  --output checkpoints/my_kg_dpo
```

Both scripts write a `training_manifest.json` to the output directory on completion.

### Step 10 — Evaluate and compare

```bash
# Evaluate the zero-shot base model
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/base/gpt2-small \
  --eval-dataset data/datasets/eval/eval_v0.1.jsonl \
  --output reports/eval_base.json

# Evaluate your new SFT checkpoint
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/my_kg_sft \
  --eval-dataset data/datasets/eval/eval_v0.1.jsonl \
  --output reports/eval_my_kg_sft.json

# Compare
python scripts/benchmark_compare.py \
  --runs reports/eval_base.json reports/eval_my_kg_sft.json \
  --output reports/my_comparison.md
```

---

## 7. Configuration Reference

| File | Key fields | Purpose |
|---|---|---|
| `configs/corpus.yaml` | `id`, `source_root`, `chunking.*`, `splits.*` | Document ingestion and chunking |
| `configs/kg_extraction.yaml` | `provider`, `prompts.*`, `cache_enabled`, `concurrency.*` | LLM KG extraction |
| `configs/kg_schema.yaml` | `entity_types`, `relation_types` | Schema for validated KG |
| `configs/pretrain_dataset.yaml` | `kg_snapshot_id`, `max_records`, `max_hop_paths` | Pre-training dataset generation |
| `configs/sft_dataset.yaml` | `kg_snapshot_id`, `max_records`, `refusal_ratio` | SFT dataset generation |
| `configs/eval_dataset.yaml` | `kg_snapshot_id`, `max_records` | Evaluation dataset generation |
| `configs/model.yaml` | `tokenizer_path`, `lora.*`, `lora.target_modules` | LoRA adapter configuration |
| `configs/sft.yaml` | `num_epochs`, `batch_size`, `learning_rate`, `max_seq_length` | SFT training hyperparameters |
| `configs/dpo.yaml` | `num_epochs`, `batch_size`, `learning_rate`, `beta` | DPO training hyperparameters |

**API keys** are read from environment variables and must never appear in config files:

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

---

## 8. KG Grounding Across the Pipeline

The table below summarises the role of the KG at each stage and what provenance is recorded.

| Stage | KG role | Provenance recorded |
|---|---|---|
| Extraction | Propose candidate entities and relations | document ID, chunk ID, prompt hash, provider response |
| Validation | Accept, reject, or normalize candidates | schema rule, validation status, evidence span |
| Dataset generation | Derive training, preference, and eval examples | KG snapshot ID, entity IDs, relation IDs, graph path IDs |
| SFT | Graph-grounded instruction-response pairs | dataset manifest, KG snapshot, graph evidence fields |
| DPO | Graph-consistent chosen / graph-inconsistent rejected pairs | preference record, checked facts, corruption type |
| Evaluation | Graph-supported token F1 and exact match | benchmark version, held-out KG facts, expected evidence |

Every generated example traces back to a specific validated graph fact and, through that fact, to a source document and chunk.

---

## 9. Benchmark Results

Evaluated on `gpt2-small` fine-tuned with progressively richer KG grounding. Metric: token F1 on single-hop QA (34 eval records drawn from the validated KG).

| Baseline | Description | Avg F1 | Δ vs B0 |
|---|---|---:|---:|
| B0 | Zero-shot (frozen gpt2-small) | 0.1033 | — |
| B1 | Raw-text SFT (no KG) | 0.0674 | −0.0359 |
| B2 | Graph-linearized continued pre-training | 0.0893 | −0.0140 |
| B3 | Graph-verbalized continued pre-training | 0.2022 | +0.0989 |
| B4 | KG-grounded SFT | **0.2184** | **+0.1151** |
| B5 | KG-grounded DPO (on B4) | 0.1129 | +0.0096 |

Key observations:

- Raw-text SFT without KG grounding (B1) degrades below the zero-shot baseline — the model overfits surface patterns without factual anchoring.
- Verbalized pre-training (B3) already recovers and improves, because natural-language KG sentences teach the model domain facts in fluent form.
- KG-grounded SFT (B4) achieves the largest gain (+0.1151 F1) by coupling graph context directly to question-answer pairs.
- DPO (B5) with the small eval set shows a modest gain over B0 but regresses relative to B4, suggesting the preference dataset needs more records or the `beta` needs tuning for this model size.

Full per-record reports are in `reports/`.

---

## 10. Testing

**C++ unit and integration tests**

```bash
cd build && ctest --output-on-failure
```

Tests cover corpus ingestion, extraction parsing, KG normalization and validation, dataset builders, and prompt rendering. The mock extraction provider uses fixtures in `tests/fixtures/` — no live API calls are required.

**Python script tests**

```bash
pytest tests/test_scripts.py -v
```

To run extraction tests against the mock provider without an API key:

```bash
# in configs/kg_extraction.yaml, set:
#   provider: mock
#   test_mode: true
./build/slmkg kg extract --config configs/kg_extraction.yaml
```
