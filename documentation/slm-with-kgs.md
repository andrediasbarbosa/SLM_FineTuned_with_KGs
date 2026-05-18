# SLM with KGs  
## Technical Specification for a KG-grounded fine-tuning of small pre-trained language models

**Version:** 0.9  
**Status:** Refined technical specification  
**Primary implementation language:** C++  
**Optional later support:** Python bindings, notebooks, evaluation dashboards, visualization utilities
---

# 1. Objective

Create a technical specification for constructing a **small language model** that uses **knowledge graphs** as a core design component throughout the full model development lifecycle.

The knowledge graphs will be extracted from a corpus of input documents using configurable external LLM API calls. The user will define both the **system prompt** and the **user prompt** used for extraction. The extraction layer must therefore be prompt-configurable, provider-configurable, reproducible, testable, and fully auditable.

The project investigates how structured knowledge can improve:

- factual consistency,
- domain grounding,
- reasoning behavior,
- sample efficiency,
- hallucination resistance,
- alignment,
- traceability,
- and controllability in compact language models.

The KG is not treated merely as an inference-time retrieval system. Instead, it is used across:

1. corpus ingestion,
2. document parsing,
3. KG extraction,
4. KG validation,
5. pre-training data generation,
6. supervised fine-tuning,
7. reinforcement learning,
8. evaluation,
9. benchmarking,
10. and optional inference-time grounding.

---

# 2. Project Scope

## 2.1 In Scope

This specification covers the design and implementation of a KG-supported SLM pipeline, including:

- Corpus ingestion from input documents.
- Chunking, cleaning, deduplication, and manifest generation.
- External LLM-based KG extraction from document chunks.
- User-defined system prompts and user prompts for extraction.
- Provider abstraction for OpenAI, Gemini, and later providers.
- Prompt versioning, hashing, provenance, and replayability.
- KG schema definition and validation.
- Entity and relation extraction.
- Entity normalization, alias handling, and deduplication.
- Relation validation and provenance tracking.
- Versioned KG snapshots.
- Graph-derived pre-training datasets.
- Graph-grounded supervised fine-tuning datasets.
- KG-based reward functions for reinforcement learning.
- Evaluation of factuality, KG consistency, reasoning, refusal behavior, and hallucination rate.
- CLI workflows for reproducible experimentation.
- Unit testing, integration testing, replay testing, golden-file testing, and benchmark testing.
- C++ core implementation with optional later Python support.

## 2.2 Out of Scope for Initial Version

The first implementation does not need to include:

- distributed training,
- full production model serving,
- online graph updates during inference,
- real-time streaming KG extraction,
- full OWL/RDF reasoning,
- large-scale graph database deployment,
- enterprise access-control integration,
- GPU kernel-level optimization.

These may be added later as separate architectural extensions.

---

# 3. Core Principle

The core design principle is:

> The KG is extracted from the input corpus, validated, versioned, and then used as a structural substrate for training, evaluation, and alignment of the SLM.

The KG is therefore not a decorative sidecar. It is a scaffold, a measuring instrument, and a training signal.

## 3.1 Knowledge Graph Grounding Strategy

Knowledge graph grounding is the central mechanism being studied in this project. The KG must be used as an explicit source of evidence, structure, constraints, and evaluation signals across the model-development pipeline.

Grounding means that model inputs, training targets, rewards, and evaluations should be traceable back to validated graph facts and, through those facts, to source corpus evidence. A model answer should therefore be considered graph-grounded only when the supporting entity, relation, evidence span, KG snapshot, and source document can be identified.

The project should distinguish between six grounding stages:

| Stage | Grounding Role | Required Traceability |
|---|---|---|
| KG extraction | Extract candidate entities and relations from source documents | document ID, chunk ID, prompt version, provider response |
| KG validation | accept, reject, or qualify extracted graph facts | schema rule, validation status, evidence span |
| Dataset generation | create graph-derived training and evaluation records | KG snapshot, entity IDs, relation IDs, graph path IDs |
| Fine-tuning | train the SLM on KG-grounded examples | dataset manifest, KG snapshot, graph evidence fields |
| Reinforcement learning | reward or penalize outputs based on graph consistency | reward record, checked facts, contradiction evidence |
| Evaluation | measure factuality, reasoning, refusal, and hallucination behavior | benchmark version, held-out facts, expected graph evidence |

### 3.1.1 Corpus-Grounded KG Extraction

The first grounding point is the extraction step. External LLM APIs may propose entities and relations, but the KG must remain grounded in the input corpus rather than in the external model's general knowledge.

Extraction requirements:

- Every extracted entity and relation must reference the source document and chunk.
- Every relation should include an evidence span or evidence quote where possible.
- Provider outputs must be stored so extraction can be replayed and audited.
- Prompt hashes must be stored with extraction records.
- Candidate facts without evidence should be marked as low confidence or rejected, depending on schema rules.

### 3.1.2 Validation-Grounded Knowledge Graphs

The validated KG is the authoritative grounding artifact for downstream training and evaluation. Raw extraction output must not be used directly for model training.

Validation requirements:

- Entity types must match the KG schema.
- Relation types must match the KG schema.
- Relation endpoints must reference canonical entities.
- Duplicate entities and aliases must be normalized.
- Rejected facts must be stored separately for audit and error analysis.
- Each accepted fact must include provenance and validation status.

### 3.1.3 Training-Data Grounding

KG grounding enters the language model primarily through generated datasets.

Training examples may include:

- graph-linearized triples,
- natural-language verbalizations of graph facts,
- evidence-grounded factual statements,
- question-answer pairs generated from graph facts,
- multi-hop reasoning examples generated from graph paths,
- contradiction examples generated from invalid or corrupted triples,
- refusal examples where the KG does not contain sufficient evidence.

Each generated example must record the KG facts used to create it. This allows later analysis of whether the model learned graph-supported behavior or merely memorized surface text.

### 3.1.4 Fine-Tuning Grounding

During supervised fine-tuning, grounding means that the target response should be supported by KG evidence.

Fine-tuning requirements:

- Answers should include only facts supported by the selected KG snapshot.
- Multi-hop answers should reference valid graph paths.
- Missing-evidence examples should teach the model to qualify or refuse unsupported claims.
- Training manifests must record the dataset version and KG snapshot.
- KG-free baselines must be kept separate from KG-grounded fine-tuning runs.

### 3.1.5 Reinforcement-Learning Grounding

During reinforcement learning or preference optimization, the KG is used as a feedback mechanism.

KG-grounded reward signals may:

- reward answers that match validated graph facts,
- reward correct relation explanations,
- reward valid multi-hop reasoning chains,
- reward appropriate uncertainty when graph evidence is missing,
- penalize unsupported factual claims,
- penalize contradictions with validated graph facts,
- penalize invented entities, relations, or paths.

The reward record must identify which KG facts were checked and which graph constraints affected the score.

### 3.1.6 Evaluation Grounding

Evaluation must test whether KG grounding improves model behavior beyond ordinary fine-tuning.

Required evaluation comparisons:

- zero-shot pre-trained checkpoint,
- KG-free fine-tuned baseline,
- graph-text continued training,
- KG-grounded supervised fine-tuning,
- KG-grounded preference optimization or RL,
- optional inference-time graph retrieval.

Evaluation should report:

- graph-supported accuracy,
- contradiction rate,
- unsupported-claim rate,
- entity-disambiguation accuracy,
- relation-understanding accuracy,
- multi-hop path correctness,
- missing-evidence refusal quality,
- general fluency impact.

### 3.1.7 Optional Inference-Time Grounding

Inference-time grounding is a later extension, not a requirement for the first implementation.

If added, inference-time grounding may retrieve graph neighborhoods and provide them as context to the model. This should be benchmarked separately from training-time grounding so the project can distinguish between knowledge internalized during fine-tuning and knowledge supplied externally at inference.

The initial implementation should therefore focus on grounding through extraction, validation, dataset generation, fine-tuning, reinforcement learning, and evaluation.

---

# 4. High-Level Architecture

    Input Documents
        |
        v
    Corpus Ingestion
        |
        v
    Text Cleaning and Chunking
        |
        v
    External LLM KG Extraction
        |
        |-- user-defined system prompt
        |-- user-defined user prompt
        |-- provider configuration
        |-- response schema validation
        |-- caching and replay
        v
    Raw KG Candidates
        |
        v
    KG Normalization
        |
        v
    KG Validation
        |
        v
    Versioned KG Snapshot
        |
        |-----------------------------|
        |                             |
        v                             v
    Dataset Generation             Evaluation Benchmark Generation
        |                             |
        v                             v
    Pre-training / SFT / RL        KG Consistency Tests
        |                             |
        v                             v
    Small Language Model           Benchmark Reports

---

# 5. Corpus Input Specification

## 5.1 Corpus Role

The corpus is the authoritative source from which the initial knowledge graph is extracted.

The system must support the following pipeline:

    raw documents
        -> parsed text
        -> normalized document records
        -> text chunks
        -> LLM extraction prompts
        -> candidate entities and relations
        -> validated KG

## 5.2 Supported Input Document Types

Initial supported formats:

- .txt
- .md
- .jsonl
- .csv
- .tsv

Later supported formats:

- .pdf
- .docx
- .html
- .xml
- .json
- .pptx
- .xlsx

The initial implementation should focus on plain text, Markdown, and JSONL, because these are easier to test and reproduce.

## 5.3 Corpus Directory Layout

Recommended project layout:

    project_root/
      configs/
        corpus.yaml
        kg_extraction.yaml
        model.yaml
        training.yaml
        benchmark.yaml

      prompts/
        kg_system_prompt_v0.1.txt
        kg_user_prompt_v0.1.txt
        relation_extraction_user_prompt_v0.1.txt
        entity_extraction_user_prompt_v0.1.txt

      data/
        raw/
          documents/
          text/
          tables/

        processed/
          corpus_v0.1/
            documents.jsonl
            chunks.jsonl
            corpus_manifest.yaml
            corpus_stats.json

        kg/
          kg_v0.1/
            schema.yaml
            entities.jsonl
            relations.jsonl
            aliases.jsonl
            rejected_candidates.jsonl
            validation_report.json
            manifest.yaml

        datasets/
          pretrain/
          sft/
          preference/
          eval/

      checkpoints/
      benchmarks/
      reports/
      src/
      tests/

## 5.4 Document Record

Each input document should be converted into a normalized document record.

    {
      "id": "doc_000001",
      "source_path": "data/raw/documents/example.md",
      "source_type": "markdown",
      "title": "Example Source",
      "author": null,
      "created_at": null,
      "ingested_at": "2026-05-12T00:00:00Z",
      "text_hash": "sha256:...",
      "language": "en",
      "license": "internal_research",
      "trust_level": "trusted",
      "domain": "neuro_symbolic_ai"
    }

Required fields:

- id
- source_path
- source_type
- ingested_at
- text_hash
- language
- trust_level

## 5.5 Chunk Record

Documents should be split into chunks before KG extraction.

    {
      "id": "chunk_000001",
      "document_id": "doc_000001",
      "chunk_index": 0,
      "text": "Knowledge graphs provide structured representations...",
      "char_start": 0,
      "char_end": 1200,
      "token_count": 240,
      "text_hash": "sha256:..."
    }

Required fields:

- id
- document_id
- chunk_index
- text
- char_start
- char_end
- token_count
- text_hash

## 5.6 Corpus Manifest

Every processed corpus must include a manifest.

    corpus:
      id: corpus_v0.1
      created_at: 2026-05-12T00:00:00Z
      source_root: data/raw
      document_count: 1280
      chunk_count: 18432
      tokenizer: tokenizer_v0.1
      min_chunk_chars: 300
      max_chunk_chars: 2000
      deduplication:
        enabled: true
        method: minhash
        threshold: 0.92
      splits:
        train: 0.8
        validation: 0.1
        test: 0.1
      hash: sha256:...

## 5.7 Corpus Quality Checks

The ingestion pipeline should compute:

- document count,
- chunk count,
- empty document rate,
- duplicate document rate,
- duplicate chunk rate,
- average chunk length,
- token count distribution,
- language distribution,
- source type distribution,
- trust-level distribution,
- train-validation-test leakage indicators.

Hard failures:

- malformed JSONL,
- duplicate document IDs,
- unreadable files,
- invalid encoding,
- missing text content,
- invalid manifest,
- missing required metadata.

Warnings:

- very short documents,
- very long documents,
- missing author metadata,
- missing license metadata,
- very high duplication,
- unusual character distribution,
- unsupported language.

## 5.8 Corpus Configuration File

The full `configs/corpus.yaml` schema. All fields are required unless marked optional.

```yaml
corpus:
  id: corpus_v0.1
  source_root: data/raw
  output_root: data/processed/corpus_v0.1
  encoding: utf-8                        # accepted input encoding; reject files that do not decode cleanly

  chunking:
    min_chunk_chars: 300                 # discard chunks shorter than this
    max_chunk_chars: 2000                # split chunks longer than this
    overlap_chars: 100                   # character overlap between adjacent chunks
    token_count_method: char_heuristic   # char_heuristic uses char_count / 4; only option in Phase 1

  deduplication:
    enabled: true
    method: exact_hash                   # Phase 1: SHA256 of normalized text (lowercase + strip whitespace)
                                         # minhash near-duplicate detection is deferred to a later phase;
                                         # when added, use the datasketch Python library (not implemented in C++)
    scope: chunks                        # deduplicate at chunk level; document-level dedup is additive

  splits:
    train: 0.80
    validation: 0.10
    test: 0.10
    seed: 42
    split_by: document_id                # document_id | entity_id | relation_id — document_id required for Phase 1
    # Split algorithm: compute SHA256(document_id + seed), convert first 4 bytes to uint32,
    # take modulo 100. Values 0–79 → train, 80–89 → validation, 90–99 → test.
    # This is deterministic per document_id and seed without requiring a global shuffle.

  quality:
    min_language_score: 0.80             # optional: skip documents below this langdetect confidence
    trusted_source_types: [txt, md, jsonl]
    warn_missing_author: true
    warn_missing_license: true
    fail_on_duplicate_ids: true
    fail_on_malformed_jsonl: true
    fail_on_unreadable_files: true

  manifest:
    write: true
    hash_algorithm: sha256
```

---

# 6. User-Defined Prompting for KG Extraction

## 6.1 Prompt Ownership

The user must be able to define:

- the system prompt,
- the user prompt,
- optional extraction-specific prompts,
- expected output schema instructions,
- extraction constraints,
- ontology or schema guidance,
- examples for few-shot extraction,
- provider-specific generation settings.

The implementation must not hard-code prompt text inside the C++ extraction logic.

Prompts must be loaded from external files or configuration references.

## 6.2 Prompt Files

Recommended prompt files:

    prompts/
      kg_system_prompt_v0.1.txt
      kg_user_prompt_v0.1.txt
      entity_extraction_system_prompt_v0.1.txt
      entity_extraction_user_prompt_v0.1.txt
      relation_extraction_system_prompt_v0.1.txt
      relation_extraction_user_prompt_v0.1.txt
      schema_alignment_system_prompt_v0.1.txt
      schema_alignment_user_prompt_v0.1.txt

## 6.3 Prompt Template Variables

User prompts should support template variables.

Required variables:

- {{document_id}}
- {{chunk_id}}
- {{chunk_text}}
- {{kg_schema}}
- {{allowed_entity_types}}
- {{allowed_relation_types}}
- {{known_entities}}
- {{extraction_mode}}
- {{output_schema}}

Example user prompt template:

    You are extracting a knowledge graph from a document chunk.

    Document ID:
    {{document_id}}

    Chunk ID:
    {{chunk_id}}

    Knowledge graph schema:
    {{kg_schema}}

    Allowed entity types:
    {{allowed_entity_types}}

    Allowed relation types:
    {{allowed_relation_types}}

    Text chunk:
    {{chunk_text}}

    Extract candidate entities and relations from the text.
    Return only valid JSON matching the required output schema.
    Do not invent facts that are not supported by the text.
    Include evidence spans for every relation.

## 6.4 System Prompt Template

Example system prompt:

    You are a careful knowledge graph extraction engine.
    Your task is to extract only entities and relations that are explicitly supported by the provided text.
    You must follow the schema exactly.
    You must not invent entities, relations, dates, attributes, or causal links.
    Return valid JSON only.
    Every relation must include evidence from the source text.
    If no valid entities or relations are present, return empty arrays.

This is only an example. The actual system prompt should be user-defined.

## 6.5 Prompt Metadata

Every extraction request must record prompt metadata.

    {
      "prompt_set_id": "kg_extract_promptset_v0.1",
      "system_prompt_path": "prompts/kg_system_prompt_v0.1.txt",
      "user_prompt_path": "prompts/kg_user_prompt_v0.1.txt",
      "system_prompt_hash": "sha256:...",
      "user_prompt_hash": "sha256:...",
      "rendered_prompt_hash": "sha256:...",
      "template_variables": {
        "document_id": "doc_000001",
        "chunk_id": "chunk_000001",
        "extraction_mode": "entities_and_relations"
      }
    }

## 6.6 Prompt Versioning

Prompts must be versioned because a prompt change can materially alter the KG.

Prompt versioning requirements:

- prompt files must have explicit version identifiers,
- prompt hashes must be stored,
- prompt path and hash must be recorded in extraction provenance,
- prompt changes must produce a new extraction manifest,
- KG snapshots must reference the prompt set used,
- model checkpoints must reference the KG snapshot.

## 6.7 Prompt Rendering Tests

Prompt rendering must be unit-tested.

Tests should verify:

- all required variables are substituted,
- missing variables produce clear errors,
- rendered prompts are deterministic,
- prompt hashes are stable,
- chunk text is inserted without corruption,
- schema text is inserted correctly,
- special characters do not break rendering.

Example tests:

    PromptRendererTest.SubstitutesAllVariables
    PromptRendererTest.FailsOnMissingVariable
    PromptRendererTest.HashStableForSameInput
    PromptRendererTest.PreservesChunkText

---

# 7. External LLM API Configuration

## 7.1 Supported Providers

Initial providers:

- OpenAI
- Gemini

Later possible providers:

- Anthropic
- local vLLM endpoint
- Ollama
- Azure OpenAI
- custom internal LLM gateway

## 7.2 Configuration Shape

    kg_extraction:
      provider: openai
      extraction_mode: entities_and_relations
      cache_enabled: true
      cache_dir: .cache/kg_extraction
      replay_mode: false

      prompts:
        prompt_set_id: kg_extract_promptset_v0.1
        system_prompt_path: prompts/kg_system_prompt_v0.1.txt
        user_prompt_path: prompts/kg_user_prompt_v0.1.txt

      output:
        expected_format: json
        schema_path: configs/kg_extraction_output_schema.json
        strict_validation: true

      retry:
        max_attempts: 3
        initial_backoff_ms: 500
        max_backoff_ms: 8000

      timeout:
        connect_ms: 5000          # TCP connection timeout
        request_ms: 30000         # total request timeout including LLM generation

      concurrency:
        max_concurrent_requests: 4   # keep at 2–4 for CPU-local setups to avoid rate limits

      openai:
        model: gpt-4.1-mini
        api_key_env: OPENAI_API_KEY
        temperature: 0.0
        max_output_tokens: 4096
        endpoint: null

      gemini:
        model: gemini-2.5-flash
        api_key_env: GEMINI_API_KEY
        temperature: 0.0
        max_output_tokens: 4096
        endpoint: null

## 7.3 Configuration Requirements

The configuration system must support:

- provider selection without code changes,
- model selection without code changes,
- user-defined prompt paths,
- environment variable references for secrets,
- provider-specific settings,
- retry settings,
- timeout settings,
- output schema validation,
- cache location,
- replay mode,
- extraction batch size,
- extraction concurrency limit,
- provenance metadata.

API keys must never be committed to source control.

## 7.4 Provider Interface

The C++ implementation should expose an extraction provider interface.

    class IKgExtractionProvider {
    public:
        virtual ~IKgExtractionProvider() = default;

        virtual ExtractionResponse extract(
            const RenderedPrompt& prompt,
            const ExtractionRequestMetadata& metadata,
            const ProviderConfig& config
        ) = 0;
    };

Concrete implementations:

    OpenAIKgExtractionProvider
    GeminiKgExtractionProvider
    MockKgExtractionProvider
    ReplayKgExtractionProvider

## 7.5 Mock Provider

The mock provider is mandatory for tests.

It returns deterministic responses for fixture chunks without making any network calls.

Use cases: unit tests, integration tests, CI pipelines, offline development, schema validation tests.

**Fixture registration:** At startup the mock provider scans `tests/fixtures/mock_provider/` and loads every `.json` file whose name is the SHA256 of a rendered prompt string. The loading sequence is:

1. Compute the SHA256 of the rendered prompt string (same hash used by `prompt_hash.hpp`).
2. Look for `tests/fixtures/mock_provider/{sha256}.json`.
3. If found, return its contents as the provider response with `status: cached`.
4. If not found, return `{"entities": [], "relations": []}` with `status: success` and log a `WARN` message naming the missing fixture hash. **Do not error** — returning empty results allows the pipeline to proceed so that downstream stages can be tested even without exhaustive fixture coverage.

**Error-path fixture:** To test provider failure handling, create a fixture file named `tests/fixtures/mock_provider/__error__{sha256}.json` with content `{"status": "provider_error", "message": "mock injected error"}`. The mock provider returns this response when a prompt hash is prefixed with `__error__` in the fixture filename.

**Generating fixture files:** Run once with the live provider and the `--record-to` flag to produce fixture files:

    slmkg kg extract \
      --corpus tests/fixtures/fixture_corpus \
      --schema tests/fixtures/kg_schema.yaml \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt \
      --output /tmp/extract_tmp \
      --config configs/kg_extraction.yaml \
      --record-to tests/fixtures/mock_provider/

## 7.6 Replay Provider

The replay provider is mandatory for reproducibility.

It should load previously recorded API responses from disk and replay them without making external calls.

Use cases:

- regression tests,
- benchmark reproduction,
- debugging extraction changes,
- comparing prompt versions,
- avoiding unnecessary API cost.

## 7.7 API Call Record

Every external LLM call should produce a call record.

    {
      "request_id": "extract_req_000001",
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "document_id": "doc_000001",
      "chunk_id": "chunk_000001",
      "prompt_set_id": "kg_extract_promptset_v0.1",
      "system_prompt_hash": "sha256:...",
      "user_prompt_hash": "sha256:...",
      "rendered_prompt_hash": "sha256:...",
      "request_timestamp": "2026-05-12T00:00:00Z",
      "response_timestamp": "2026-05-12T00:00:03Z",
      "temperature": 0.0,
      "max_output_tokens": 4096,
      "status": "success",
      "raw_response_hash": "sha256:...",
      "parsed_response_hash": "sha256:..."
    }

Statuses:

- success
- provider_error
- timeout
- invalid_json
- schema_validation_failed
- empty_response
- retry_exhausted
- skipped
- cached
- replayed

---

# 8. KG Extraction Output Schema

## 8.1 Required Output Shape

The knowledge graph used in this project is a **hypergraph**. A hyperedge connects a non-empty list of head entities to a non-empty list of tail entities through a typed relation. A standard binary triple is the special case where both lists contain exactly one element. This model is more expressive than pairwise relations and better captures real-world assertions such as "A and B co-authored C and D" or "X, Y, and Z jointly support W".

The external LLM should return candidate entities and hyperedge relations in a strict JSON-compatible structure. Every relation field that previously accepted a single ID now accepts a non-empty array.

    {
      "entities": [
        {
          "local_id": "e1",
          "name": "Knowledge Graph",
          "type": "Concept",
          "aliases": ["KG"],
          "description": "A structured representation of entities and relations.",
          "evidence_text": "Knowledge graphs provide structured representations...",
          "confidence": 0.91
        }
      ],
      "relations": [
        {
          "head_local_ids": ["e1"],
          "relation_type": "supports",
          "tail_local_ids": ["e2", "e3"],
          "evidence_text": "Knowledge graphs support factual grounding and reasoning...",
          "confidence": 0.87
        }
      ]
    }

## 8.2 Output Validation

The extraction parser must validate:

- valid JSON,
- required top-level fields,
- entity array type,
- relation array type,
- required entity fields,
- required relation fields,
- allowed entity types,
- allowed relation types,
- valid confidence range,
- `head_local_ids` is a non-empty array,
- `tail_local_ids` is a non-empty array,
- all IDs in `head_local_ids` and `tail_local_ids` exist in the entities array,
- all head entities match the relation's schema `domain` type,
- all tail entities match the relation's schema `range` type,
- `|head_local_ids| >= head_arity.min` and `<= head_arity.max` (if max is set),
- `|tail_local_ids| >= tail_arity.min` and `<= tail_arity.max` (if max is set),
- evidence text is non-empty,
- evidence text appears in or is traceable to source chunk,
- no unsupported relation types,
- no malformed identifiers.

Invalid outputs should not enter the KG directly. They should be stored in rejected candidate logs.

## 8.3 Candidate Entity Record

    {
      "candidate_id": "cand_ent_000001",
      "document_id": "doc_000001",
      "chunk_id": "chunk_000001",
      "name": "Knowledge Graph",
      "type": "Concept",
      "aliases": ["KG"],
      "description": "A structured representation of entities and relations.",
      "evidence_text": "Knowledge graphs provide structured representations...",
      "confidence": 0.91,
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "prompt_set_id": "kg_extract_promptset_v0.1",
      "validation_status": "unvalidated"
    }

## 8.4 Candidate Relation Record

    {
      "candidate_id": "cand_rel_000001",
      "document_id": "doc_000001",
      "chunk_id": "chunk_000001",
      "head_candidate_ids": ["cand_ent_000001"],
      "relation_type": "supports",
      "tail_candidate_ids": ["cand_ent_000002", "cand_ent_000003"],
      "evidence_text": "Knowledge graphs support factual grounding...",
      "confidence": 0.87,
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "prompt_set_id": "kg_extract_promptset_v0.1",
      "validation_status": "unvalidated"
    }

## 8.5 Evidence Provenance Enrichment

The LLM returns a flat `evidence_text` string per relation. The canonical relation record (Section 9.3) stores a structured `evidence` array. The response parser is responsible for this transform using the request metadata — the document and chunk provenance comes from the extraction request, not from the LLM.

**Transform rule:**

For every validated relation extracted from a single-chunk request:

```
evidence = [
  {
    "source_id":  <document_id from ExtractionRequestMetadata>,
    "chunk_id":   <chunk_id from ExtractionRequestMetadata>,
    "text":       <evidence_text string from LLM response>
  }
]
```

For batch requests where multiple chunks are submitted together, the parser must match each relation to its source chunk using the `chunk_id` carried in the request batch entry. Relations whose `evidence_text` cannot be traced to any chunk in the batch must be rejected and logged to `rejected_candidates.jsonl`.

The original `evidence_text` string must be preserved verbatim in the candidate record. Evidence enrichment into the structured array happens only after validation passes and the candidate is promoted to a canonical record.

---

# 9. KG Construction

## 9.1 KG Schema

The project should define a schema before extraction.

The schema should include:

- entity types,
- relation types,
- attribute types,
- domain constraints,
- range constraints,
- hyperedge arity (minimum and maximum head and tail counts per relation type),
- allowed qualifiers,
- evidence requirements,
- confidence semantics,
- validation rules.

Example schema:

    schema_version: kg_schema_v0.1

    entity_types:
      - name: Person
        required_fields: [id, canonical_name]
      - name: Organization
        required_fields: [id, canonical_name]
      - name: Concept
        required_fields: [id, canonical_name]
      - name: Document
        required_fields: [id, title, source_uri]

    relation_types:
      - name: authored
        domain: Person
        range: Document
        head_arity: {min: 1, max: null}   # one or more persons
        tail_arity: {min: 1, max: null}   # one or more documents

      - name: member_of
        domain: Person
        range: Organization
        head_arity: {min: 1, max: null}
        tail_arity: {min: 1, max: 1}      # a person belongs to one org per hyperedge

      - name: defines
        domain: Document
        range: Concept
        head_arity: {min: 1, max: 1}
        tail_arity: {min: 1, max: null}   # a document may define many concepts

      - name: supports
        domain: Concept
        range: Concept
        head_arity: {min: 1, max: null}
        tail_arity: {min: 1, max: null}

      - name: contradicts
        domain: Concept
        range: Concept
        head_arity: {min: 1, max: null}
        tail_arity: {min: 1, max: null}

## 9.2 Canonical Entity Record

After normalization, candidate entities should become canonical entities.

    {
      "id": "ent_000001",
      "type": "Concept",
      "canonical_name": "Knowledge Graph",
      "aliases": ["KG", "semantic graph"],
      "description": "A structured representation of entities and relations.",
      "source_ids": ["doc_000001", "doc_000017"],
      "chunk_ids": ["chunk_000001", "chunk_000095"],
      "confidence": 0.94,
      "validation_status": "auto_validated",
      "created_at": "2026-05-12T00:00:00Z",
      "updated_at": "2026-05-12T00:00:00Z"
    }

## 9.3 Canonical Relation Record

    {
      "id": "rel_000042",
      "head_ids": ["ent_000001"],
      "relation_type": "supports",
      "tail_ids": ["ent_000017", "ent_000031"],
      "qualifiers": {
        "context": "neuro-symbolic AI",
        "scope": "domain grounding"
      },
      "evidence": [
        {
          "source_id": "doc_000001",
          "chunk_id": "chunk_000001",
          "text": "Knowledge graphs support factual grounding..."
        }
      ],
      "confidence": 0.88,
      "extraction_provider": "openai",
      "extraction_model": "gpt-4.1-mini",
      "prompt_set_id": "kg_extract_promptset_v0.1",
      "system_prompt_hash": "sha256:...",
      "user_prompt_hash": "sha256:...",
      "created_at": "2026-05-12T00:00:00Z",
      "validation_status": "auto_validated"
    }

## 9.4 Validation Status Values

Allowed values:

    unvalidated
    auto_validated
    auto_rejected
    human_validated
    human_rejected
    deprecated

## 9.5 KG Versioning

Each KG snapshot must include a manifest.

    kg_snapshot:
      id: kg_domain_v0.1
      created_at: 2026-05-12T00:00:00Z
      schema_version: kg_schema_v0.1
      corpus_version: corpus_v0.1
      entity_count: 18420
      relation_count: 92312
      prompt_set_id: kg_extract_promptset_v0.1
      system_prompt_hash: sha256:...
      user_prompt_hash: sha256:...
      extraction_config_hash: sha256:...
      normalization_config_hash: sha256:...
      validation_config_hash: sha256:...
      previous_snapshot: null

Model checkpoints must reference the KG snapshot used during training or evaluation.

---

# 10. Dataset Generation

## 10.1 Dataset Types

The project should generate:

- pre-training datasets,
- supervised fine-tuning datasets,
- preference datasets,
- reinforcement learning datasets,
- evaluation datasets,
- benchmark datasets.

## 10.2 Pre-Training Data

KG-derived pre-training samples may include:

### Triple Linearization

    [ENTITY] Paris [RELATION] capital_of [ENTITY] France

### Natural Language Verbalization

    Paris is the capital of France.

### Evidence-Grounded Statement

    According to document doc_0042, Paris is described as the capital of France.

### Path Linearization

    A --regulates--> B --causes--> C

### Multi-Hop Explanation

    A may influence C because A regulates B, and B causes C.

### Negative Sample

    Claim: Paris is the capital of Germany.
    Label: contradicted_by_graph.
    Evidence: Paris capital_of France.

### Hyperedge Linearization

When a relation has multiple head or tail entities, all participants are included in the linearized form.

Single head, multiple tails:

    [ENTITY] Knowledge Graph [RELATION] supports [ENTITY] Factual Grounding [ENTITY] Reasoning

Multiple heads, single tail:

    [ENTITY] Einstein [ENTITY] Bohr [RELATION] debated [ENTITY] Quantum Mechanics

Multiple heads, multiple tails:

    [ENTITY] Einstein [ENTITY] Bohr [RELATION] contributed_to [ENTITY] Special Relativity [ENTITY] General Relativity

Natural language verbalization of hyperedges:

    Einstein and Bohr contributed to Special Relativity and General Relativity.

Evidence-grounded hyperedge:

    According to document doc_0042, Einstein and Bohr are described as contributors to Special Relativity and General Relativity.

For the binary special case (`|head_ids| == 1` and `|tail_ids| == 1`), the standard triple linearization format is used without modification.

## 10.2a Negative Sample Generation Algorithm

Negative samples are used in pre-training contradiction examples and in preference dataset rejected outputs. The following algorithm must be used so that no generated negative is accidentally a true KG fact.

**Inputs:** validated KG snapshot (set of hyperedges), KG schema (entity types, relation types, arity constraints), random seed.

**For each positive hyperedge `(H, r, T)` where `H = [h1, h2, ...]` and `T = [t1, t2, ...]`:**

1. Choose corruption mode uniformly at random from the applicable set:
   - `corrupt_tail_entity` — replace one tail entity with a random wrong entity (always available if `|T| >= 1`)
   - `corrupt_relation` — replace the relation type with a wrong one (always available if alternatives exist)
   - `shrink_tails` — remove one tail entity (only if `|T| > tail_arity.min`)
   - `expand_tails` — add a random wrong entity to the tail list (only if `|T| < tail_arity.max` or max is null)
   - `shrink_heads` — remove one head entity (only if `|H| > head_arity.min`)
   - Modes that are not applicable given the current arity are excluded from the random selection.

2. **Corrupt tail entity:**
   - Sample a random entity `e′` from the KG where `type(e′) == schema.range(r)` and `e′ ∉ T`.
   - Replace one randomly chosen element of `T` with `e′`.

3. **Corrupt relation:**
   - Collect all relation types `R′` where `schema.domain(R′)` is compatible with all entities in `H`.
   - Sample `r′` from `R′` where `r′ ≠ r`. Fall back to `corrupt_tail_entity` if no alternative exists.

4. **Shrink tails:** remove one randomly chosen element from `T`. The resulting `T′` must still satisfy `|T′| >= tail_arity.min`.

5. **Expand tails:** add one randomly sampled entity `e′` where `type(e′) == schema.range(r)` and `e′ ∉ T`.

6. **Shrink heads:** remove one randomly chosen element from `H`. Only applied when `|H| > 1` and `|H| > head_arity.min`.

7. **Collision check:** verify the corrupted hyperedge `(H′, r′, T′)` is NOT present in the validated KG.
   - Collision is defined as: there exists a validated hyperedge with the same `relation_type`, `head_ids` (as a set), and `tail_ids` (as a set).
   - If it IS present, retry up to 10 times with a fresh random sample.
   - If still colliding after 10 retries, skip this hyperedge and log to `rejected_negatives.jsonl`.

8. Write the accepted negative sample with fields:
   - `negative_type`: one of the corruption mode names above
   - `source_relation_id`: the ID of the positive hyperedge
   - `linearized`: the corrupted hyperedge in the same format used for positive hyperedges

## 10.3 Pre-Training Mixture

Recommended configurable mixture:

    pretraining_mixture:
      natural_text: 0.70
      graph_linearized: 0.10
      graph_verbalized: 0.10
      graph_reasoning_paths: 0.05
      graph_contradiction_samples: 0.05

The mixture must be logged in the dataset manifest and training manifest.

## 10.4 Supervised Fine-Tuning Tasks

Required SFT task types:

- single_hop_qa
- multi_hop_qa
- entity_disambiguation
- relation_explanation
- claim_verification
- contradiction_detection
- missing_evidence_refusal
- graph_path_explanation

Example SFT record:

    {
      "id": "sft_000001",
      "task_type": "single_hop_qa",
      "instruction": "Answer using the provided graph evidence.",
      "input": "What is the capital of France?",
      "graph_context": [
        "Paris capital_of France"
      ],
      "output": "Paris is the capital of France.",
      "evidence_relation_ids": ["rel_000042"],
      "kg_snapshot": "kg_domain_v0.1"
    }

## 10.5 Missing Evidence Refusal Example

    {
      "id": "sft_000002",
      "task_type": "missing_evidence_refusal",
      "instruction": "Answer only if the graph supports the claim.",
      "input": "Is X the founder of Y?",
      "graph_context": [],
      "output": "The available graph evidence does not support that claim.",
      "kg_snapshot": "kg_domain_v0.1"
    }

## 10.6 Preference Dataset

Preference records should compare graph-consistent and graph-inconsistent answers.

    {
      "id": "pref_000001",
      "prompt": "Explain the relationship between A and C.",
      "chosen": "A may influence C through B, because A regulates B and B causes C.",
      "rejected": "A directly causes C.",
      "reason": "The rejected answer collapses a two-hop path into an unsupported direct relation.",
      "evidence_relation_ids": ["rel_000001", "rel_000002"],
      "kg_snapshot": "kg_domain_v0.1"
    }

## 10.7 Data Requirements for Pre-Training, Fine-Tuning, and RL

The project must define separate data requirements for each training stage. These stages use different data formats, quality checks, sampling rules, and traceability requirements.

The term "pre-training" in this project refers to continued pre-training or domain-adaptive language modeling from a pre-trained checkpoint, not training a language model from random initialization.

### 10.7.1 Common Data Requirements

All generated datasets must satisfy the following requirements:

- every record must have a stable unique ID,
- every record must identify its source dataset version,
- every graph-derived record must reference the KG snapshot used to create it,
- every graph-derived record must reference the entity IDs, relation IDs, or graph path IDs used as evidence,
- every record derived from corpus text must reference document IDs and chunk IDs,
- every generated record must record the generator version and configuration hash,
- train, validation, and test splits must be generated before model training,
- split membership must be stable across repeated runs with the same seed,
- benchmark and held-out evaluation records must never leak into training data,
- dataset manifests must include record counts, token counts, source counts, KG snapshot IDs, prompt set IDs, and content hashes.

Recommended split policy:

    dataset_splits:
      train: 0.80
      validation: 0.10
      test: 0.10
      split_seed: 42
      split_by:
        - document_id
        - entity_id
        - relation_id
        - graph_path_id

Splitting only by individual example is not sufficient. The project must support stricter split modes that hold out whole documents, entities, relations, or graph paths so that evaluation can measure generalization rather than memorization.

### 10.7.2 Continued Pre-Training Data Requirements

Continued pre-training data should teach the model domain language and graph-aware textual structure while preserving general fluency.

Required data sources:

- cleaned natural text chunks from the corpus,
- graph-linearized triples,
- natural-language verbalizations of graph facts,
- evidence-grounded factual statements,
- graph path linearizations,
- multi-hop graph explanations,
- controlled contradiction samples,
- optional schema descriptions and relation definitions.

Minimum record fields:

    {
      "id": "pretrain_000001",
      "record_type": "graph_verbalized",
      "text": "Paris is the capital of France.",
      "source_document_ids": ["doc_0042"],
      "source_chunk_ids": ["chunk_000133"],
      "evidence_entity_ids": ["ent_paris", "ent_france"],
      "evidence_relation_ids": ["rel_000042"],
      "kg_snapshot": "kg_domain_v0.1",
      "generator": "graph_verbalizer_v0.1",
      "token_count": 8
    }

Quality requirements:

- natural text must be cleaned, deduplicated, and chunked consistently,
- graph facts must come only from validated KG snapshots,
- graph-derived text must preserve relation direction,
- negative or contradiction samples must be clearly labeled and kept out of plain causal language modeling unless the format makes the contradiction explicit,
- graph-derived examples should be paraphrased to avoid overfitting to one serialization format,
- examples should fit within the selected model context length,
- token counts must be computed with the active tokenizer.

Recommended starting volume:

| Dataset Scale | Natural Text Records | KG-Derived Records | Use Case |
|---|---:|---:|---|
| Fixture | 20 to 100 | 20 to 100 | Unit tests and golden-file tests |
| Development | 1k to 10k | 1k to 10k | Local debugging and ablations |
| Study run | 50k to 250k | 25k to 150k | Main experimental comparisons |

The first local implementation should use fixture and development scales. Larger study runs should be introduced only after extraction, validation, and leakage checks are stable.

### 10.7.3 Supervised Fine-Tuning Data Requirements

SFT data should teach instruction-following behavior grounded in graph evidence.

Required task families:

- single-hop factual QA,
- multi-hop factual QA,
- entity disambiguation,
- relation explanation,
- claim verification,
- contradiction detection,
- missing-evidence refusal,
- graph path explanation,
- evidence citation or evidence summarization where appropriate.

Minimum record fields:

    {
      "id": "sft_000001",
      "task_type": "single_hop_qa",
      "instruction": "Answer using only the provided graph evidence.",
      "input": "What is the capital of France?",
      "graph_context": ["Paris capital_of France"],
      "output": "Paris is the capital of France.",
      "source_document_ids": ["doc_0042"],
      "source_chunk_ids": ["chunk_000133"],
      "evidence_entity_ids": ["ent_paris", "ent_france"],
      "evidence_relation_ids": ["rel_000042"],
      "kg_snapshot": "kg_domain_v0.1",
      "split": "train"
    }

Quality requirements:

- outputs must be fully supported by graph evidence,
- graph context must include enough evidence to answer the instruction,
- missing-evidence examples must not accidentally include supporting evidence in the prompt,
- multi-hop examples must include valid graph paths rather than inferred shortcuts,
- entity-disambiguation examples should include plausible aliases or ambiguous names,
- relation-explanation examples should include relation direction and type,
- unsupported claims must teach qualified refusal rather than generic refusal,
- SFT records must be balanced across task types where possible.

Recommended starting volume:

| Dataset Scale | Records | Notes |
|---|---:|---|
| Fixture | 20 to 50 | Hand-check every record |
| Development | 500 to 2,000 | Enough to validate training scripts |
| Study run | 5,000 to 50,000 | Enough for ablations with GPT-2 Small |

The SFT dataset should include both graph-context-present and graph-context-absent examples. This is necessary to evaluate whether the model learns to rely on evidence rather than always answering from prior knowledge.

### 10.7.4 Preference and RL Data Requirements

Preference and RL datasets should train or evaluate whether the model prefers graph-consistent responses over graph-inconsistent responses.

Required data sources:

- SFT prompts reused with alternative candidate answers,
- graph-consistent chosen answers,
- graph-inconsistent rejected answers,
- unsupported answers,
- overconfident answers when evidence is missing,
- answers with invalid graph paths,
- answers with correct facts but wrong relation direction,
- answers that are fluent but not supported by the KG.

Minimum preference record fields:

    {
      "id": "pref_000001",
      "prompt": "Explain the relationship between A and C.",
      "graph_context": [
        "A regulates B",
        "B causes C"
      ],
      "chosen": "A may influence C through B, because A regulates B and B causes C.",
      "rejected": "A directly causes C.",
      "preference_reason": "The rejected answer invents an unsupported direct relation.",
      "reward_tags": ["multi_hop_correct", "unsupported_direct_relation"],
      "evidence_relation_ids": ["rel_000001", "rel_000002"],
      "kg_snapshot": "kg_domain_v0.1",
      "split": "train"
    }

Reward-model or RL records should additionally support:

- scalar reward values,
- component reward values,
- facts checked by the reward function,
- contradiction evidence,
- refusal correctness,
- language-quality score,
- reward function version.

Example reward record:

    {
      "id": "reward_000001",
      "prompt_id": "pref_000001",
      "candidate": "A directly causes C.",
      "reward_total": -0.65,
      "reward_components": {
        "fact": -0.4,
        "consistency": -0.4,
        "reasoning": -0.2,
        "language": 0.35
      },
      "checked_relation_ids": ["rel_000001", "rel_000002"],
      "failure_type": "unsupported_direct_relation",
      "reward_function_version": "kg_reward_v0.1",
      "kg_snapshot": "kg_domain_v0.1"
    }

Quality requirements:

- chosen and rejected answers should be similar in length and fluency where possible,
- rejected answers should represent realistic model errors, not only trivial corruptions,
- preference pairs must be balanced across error types,
- reward functions must be tested on held-out examples before RL training,
- RL evaluation must use held-out entities, relations, or graph paths,
- reward datasets must include missing-evidence cases so the model can learn calibrated uncertainty,
- reward records must distinguish factual correctness from language quality.

Recommended starting volume:

| Dataset Scale | Preference Pairs | Reward Records | Use Case |
|---|---:|---:|---|
| Fixture | 10 to 30 | 10 to 30 | Reward function tests |
| Development | 200 to 1,000 | 500 to 2,000 | DPO/PPO smoke tests |
| Study run | 2,000 to 20,000 | 5,000 to 50,000 | Main RL or preference optimization experiments |

For the first implementation, DPO-style preference optimization is preferred over PPO because it is simpler, more stable, and easier to run locally.

### 10.7.5 Data Leakage and Contamination Controls

The project must include explicit leakage controls because KG-derived examples can easily leak evaluation facts into training.

Required controls:

- hold out benchmark facts before generating training examples,
- support document-level, entity-level, relation-level, and path-level split policies,
- ensure negative samples do not contradict facts reserved for evaluation,
- ensure paraphrases of held-out facts are not present in training,
- store split decisions in dataset manifests,
- provide a command that audits overlap between train, validation, test, and benchmark sets.

The benchmark set should be generated from a reserved KG slice where possible. This allows evaluation to test whether the model generalizes graph reasoning patterns rather than memorizing specific graph statements.

### 10.7.6 Dataset Manifest Requirements

Every dataset build must produce a manifest.

Required manifest fields:

    dataset_manifest:
      id: sft_v0.1
      created_at: 2026-05-12T00:00:00Z
      dataset_type: sft
      generator_version: sft_builder_v0.1
      config_hash: sha256:...
      corpus_version: corpus_v0.1
      kg_snapshot: kg_domain_v0.1
      prompt_set_id: kg_extract_promptset_v0.1
      tokenizer: openai-community/gpt2
      split_policy: entity_and_path_holdout
      counts:
        total_records: 10000
        train_records: 8000
        validation_records: 1000
        test_records: 1000
      token_counts:
        train_tokens: 1200000
        validation_tokens: 150000
        test_tokens: 150000
      task_distribution:
        single_hop_qa: 2000
        multi_hop_qa: 2000
        entity_disambiguation: 1000
        relation_explanation: 1000
        claim_verification: 1500
        contradiction_detection: 1500
        missing_evidence_refusal: 1000
      hashes:
        train: sha256:...
        validation: sha256:...
        test: sha256:...

Training must fail early if the dataset manifest is missing, incompatible with the selected KG snapshot, or incompatible with the selected tokenizer.

## 10.8 Data Sources and LLM-Assisted Data Generation

The LoRA fine-tuning stage depends on supervised examples that teach the checkpoint how to answer, reason, refuse, and explain using KG evidence. These examples can come from several sources, but every graph-grounded example must remain traceable to the validated KG and the original corpus.

### 10.8.1 Required Data for LoRA Fine-Tuning

LoRA fine-tuning requires instruction-response records in JSONL format. The C++ dataset builders should produce these records from the validated KG snapshot and corpus metadata.

Required LoRA SFT inputs:

- a pre-trained checkpoint, such as `openai-community/gpt2`,
- the matching tokenizer,
- KG structural special tokens,
- an SFT JSONL dataset,
- a validation JSONL dataset,
- a dataset manifest,
- the KG snapshot used to generate the dataset,
- the training configuration,
- the LoRA configuration.

Required SFT record content:

- instruction text,
- user input or question,
- optional graph context,
- target output,
- task type,
- source document IDs,
- source chunk IDs,
- evidence entity IDs,
- evidence relation IDs,
- KG snapshot ID,
- split label.

The model is trained on formatted text derived from each record. For example:

    Instruction: Answer using only the provided graph evidence.

    Graph:
    <kg>
    Paris capital_of France
    </kg>

    Question:
    What is the capital of France?

    Answer:
    Paris is the capital of France.

The target loss should be applied primarily to the answer portion when using instruction-style SFT. This prevents the model from being optimized to merely reproduce the prompt text.

### 10.8.2 Primary Data Sources

The project should treat the source corpus as the primary authority.

Primary data sources:

| Source | Used For | Notes |
|---|---|---|
| Raw corpus documents | natural text, evidence spans, domain language | must be stored locally and versioned |
| Cleaned corpus chunks | continued pre-training, prompt inputs | produced by corpus ingestion |
| Validated KG snapshot | graph facts, paths, constraints | authoritative source for graph-grounded examples |
| KG schema | entity and relation constraints | controls extraction and validation |
| Prompt files | extraction and generation behavior | user-defined and hash-tracked |
| Human review notes | optional quality labels | useful for high-value evaluation sets |

The external LLM provider is not a primary factual source. It is a transformation tool that helps extract, verbalize, rewrite, or contrast information already present in the corpus or KG.

### 10.8.3 Data That Can Be Generated Directly From the KG

Many useful examples can be generated without additional LLM calls.

Deterministic KG-derived data:

- triple linearizations,
- relation direction examples,
- graph path linearizations,
- schema-constrained negative triples,
- missing-evidence refusal prompts,
- entity alias matching examples,
- graph consistency checks,
- benchmark records from held-out graph facts.

These should be generated by C++ builders because they are easy to test and reproduce.

Example deterministic SFT generation:

    KG fact:
    Paris capital_of France

    Generated question:
    What is the capital of France?

    Generated answer:
    Paris is the capital of France.

    Evidence:
    rel_000042

This kind of generation is cheap, offline, and reproducible, but it can become formulaic. It should be mixed with paraphrased and more natural examples where possible.

### 10.8.4 Data That Can Be Generated With LLM API Calls

External LLM APIs may be used to generate richer training data, but only under strict grounding constraints.

Allowed LLM-assisted generation tasks:

- convert KG triples into natural-language paraphrases,
- create question-answer pairs from validated graph facts,
- create multi-hop reasoning explanations from validated graph paths,
- create relation explanation examples,
- create entity disambiguation examples using known aliases,
- create fluent missing-evidence refusal examples,
- create preference pairs where the chosen answer is graph-consistent and the rejected answer contains a controlled error,
- rewrite deterministic examples into varied styles while preserving graph facts.

Not allowed:

- inventing new entities not present in the corpus or KG,
- inventing new relations not present in the validated KG,
- adding unsupported facts from the provider model's general knowledge,
- generating evaluation answers without linking them to held-out KG evidence,
- using LLM-generated facts directly as KG facts without extraction, validation, and provenance.

LLM-generated examples must include:

- the input KG facts or graph path supplied to the provider,
- the prompt set ID,
- the provider name,
- the provider model,
- the rendered prompt hash,
- the raw provider response hash,
- the generated record IDs,
- the KG snapshot ID,
- validation status.

### 10.8.5 LLM-Assisted SFT Example Generation

The dataset builder may send a validated graph fact to an external LLM and ask for multiple grounded QA variants.

Example input to the provider:

    Graph evidence:
    Paris capital_of France

    Evidence relation ID:
    rel_000042

    Task:
    Generate three question-answer examples.
    Use only the graph evidence.
    Do not add unsupported facts.
    Return JSON only.

Example generated output:

    [
      {
        "question": "What is the capital of France?",
        "answer": "Paris is the capital of France."
      },
      {
        "question": "Which city is identified as France's capital?",
        "answer": "Paris is identified as France's capital."
      },
      {
        "question": "According to the graph, how is Paris related to France?",
        "answer": "Paris has the relation capital_of to France."
      }
    ]

The C++ pipeline must validate that each generated answer remains consistent with the original relation before admitting the records into the SFT dataset.

### 10.8.6 LLM-Assisted Preference Data Generation

Preference data can also be generated using provider calls, but the error modes must be controlled.

Recommended generation pattern:

- provide the validated graph evidence,
- ask the provider for one graph-consistent answer,
- ask for one rejected answer with a specified error type,
- require the provider to label the error,
- validate that the rejected answer actually contains the intended error,
- store both answers with evidence IDs and generation metadata.

Useful rejected-answer error types:

- unsupported direct relation,
- wrong relation direction,
- wrong entity,
- missing intermediate node,
- overconfident answer with missing evidence,
- fluent but unsupported claim,
- unnecessary refusal despite available evidence.

Preference examples generated by an LLM should be sampled for human review before being used in study runs.

### 10.8.7 Human-Authored and Human-Reviewed Data

Some data should be manually authored or reviewed, especially for evaluation.

Human-authored or human-reviewed data is recommended for:

- KG schema design,
- high-value benchmark questions,
- ambiguous entity disambiguation examples,
- relation definitions,
- multi-hop reasoning benchmarks,
- final held-out evaluation sets,
- prompt quality review,
- examples used in publication-quality analysis.

The project can run without human-reviewed data for early prototypes, but final claims about grounding quality should include at least a small human-reviewed evaluation slice.

### 10.8.8 Recommended Data-Sourcing Strategy

Recommended sequence:

- start with deterministic KG-derived examples,
- add LLM-paraphrased SFT examples after validation passes,
- add LLM-generated preference pairs for DPO only after SFT benchmarks are stable,
- reserve a held-out KG slice before any training data is generated,
- human-review a small benchmark subset before final reporting.

This keeps the pipeline automatable while preserving the central requirement: generated training data must remain grounded in validated KG evidence and source corpus provenance.

---

# 11. Model Integration Strategies

The project should evaluate multiple KG integration strategies.

## 11.1 Strategy A: Text-Only Graph Integration

Convert graph facts into structured text and natural language samples.

Advantages:

- simplest implementation,
- compatible with standard language modeling,
- easy to benchmark,
- useful first baseline.

Disadvantages:

- graph structure is flattened,
- graph reasoning may become memorization,
- updates require dataset regeneration or retraining.

Recommended as the first implementation.

## 11.2 Strategy B: Graph Embedding Integration

Train or import graph embeddings and combine them with token embeddings.

Advantages:

- preserves more graph structure,
- supports entity-aware representations,
- useful for compact domain models.

Disadvantages:

- more architectural complexity,
- token-entity alignment becomes difficult,
- harder to debug.

## 11.3 Strategy C: Retrieval-Augmented Graph Context

Retrieve graph neighborhoods during training or inference.

Advantages:

- KG remains externally updateable,
- reduces need to memorize all facts,
- supports explainability.

Disadvantages:

- introduces retrieval complexity,
- benchmark must distinguish retrieved knowledge from internalized knowledge.

## 11.4 Strategy D: Auxiliary Graph Objectives

Add auxiliary objectives:

- entity prediction,
- relation prediction,
- masked entity recovery,
- graph path completion,
- contradiction classification.

Advantages:

- encourages structural learning,
- provides explicit graph-specific losses.

Disadvantages:

- requires custom training loops,
- may destabilize small-model training if over-weighted.

## 11.5 Recommended Progression

    Phase 1: Text-only graph integration
    Phase 2: Graph-grounded SFT
    Phase 3: KG-based evaluation
    Phase 4: KG-based RL
    Phase 5: Graph embeddings or retrieval
    Phase 6: Hybrid architecture

## 11.6 Small Language Model Architecture

The baseline SLM should be a pre-trained checkpoint fine-tuned on the KG-derived datasets, rather than a model trained from scratch. This eliminates the need to implement a training framework, reduces Phase 5 risk, and lets the research focus on the effect of KG-derived data and rewards rather than on training dynamics of an undertrained model.

The fine-tuning implementation uses Python with the HuggingFace `transformers` library. All pipeline stages prior to training — corpus ingestion, KG extraction, dataset generation — remain in C++. The CLI wrapper (`slmkg train`) invokes the Python fine-tuning scripts.

### 11.6.1 Supported Pre-Trained Checkpoints

Two checkpoints are supported for CPU-local development. One must be selected in `configs/model.yaml`.

**Development checkpoint: GPT-2 Small**

| Property | Value |
|---|---|
| Publisher | OpenAI |
| Parameters | 124M |
| Architecture | Decoder-only, learned positional embeddings, LayerNorm, GELU |
| Context length | 1024 tokens |
| Vocabulary | 50,257 tokens |
| License | MIT |
| HuggingFace ID | `openai-community/gpt2` |

GPT-2 Small is used for development, ablations, and fast iteration. At 124M parameters it fine-tunes quickly on CPU (minutes to low tens of minutes per epoch on a modern machine with a small dataset), making it practical for the full A1–A14 ablation suite.

**Experiment checkpoint: GPT-2 Medium**

| Property | Value |
|---|---|
| Publisher | OpenAI |
| Parameters | 355M |
| Architecture | Decoder-only, learned positional embeddings, LayerNorm, GELU |
| Context length | 1024 tokens |
| Vocabulary | 50,257 tokens |
| License | MIT |
| HuggingFace ID | `openai-community/gpt2-medium` |

GPT-2 Medium is used for final benchmark runs and publication-quality comparisons (B0–B5). It is larger enough to show meaningful differences between KG integration strategies while remaining tractable on CPU with LoRA.

Both checkpoints share the same tokenizer and vocabulary, which means datasets, tokenized artifacts, and token-count estimates are identical regardless of which checkpoint is active. Neither requires a HuggingFace token or license agreement.

### 11.6.2 Implementation Language for Fine-Tuning

Fine-tuning is implemented in Python using:

- `transformers` for model loading, training loop, and tokenization.
- `datasets` for dataset ingestion from the JSONL outputs of the C++ pipeline.
- `peft` for optional LoRA adapters (reduces memory requirements without degrading research validity).
- `trl` for supervised fine-tuning (`SFTTrainer`) and preference optimization (`DPOTrainer`).
- `accelerate` for mixed-precision training and multi-GPU support.

The Python training scripts live in `scripts/` and are invoked by the `slmkg train` CLI commands. They read configurations from `configs/model.yaml` and `configs/training.yaml`.

There is no compilation step involved in fine-tuning GPT-2. The pre-trained weights are loaded from disk into memory, the optimizer updates them in-place during training, and the result is written back to disk as new weight files. The original checkpoint is never modified. Nothing is compiled, linked, or rebuilt at any stage of the fine-tuning process.

### 11.6.3 KG-Specific Token Extensions

Both GPT-2 checkpoints use the same BPE tokenizer with a 50,257-token vocabulary. Entity names and relation types should be represented as natural-language strings, not as new vocabulary tokens.

However, structural KG formatting tokens should be added to the tokenizer:

```text
<kg>
</kg>
<entity>
</entity>
<relation>
</relation>
<evidence>
</evidence>
<unknown_graph_evidence>
```

These tokens must be added via `tokenizer.add_special_tokens(...)` before fine-tuning begins. The embedding matrix must be resized accordingly (`model.resize_token_embeddings(...)`). The new token embeddings should be initialized to the mean of the existing embeddings.

Because both checkpoints share the same base tokenizer, the extended tokenizer is generated once and stored in `checkpoints/tokenizer_kg_v0.1/`. The token additions must be recorded in the training manifest so that every checkpoint is associated with the correct tokenizer version.

### 11.6.4 Fine-Tuning Configuration

Fine-tuning settings are loaded from `configs/training.yaml`.

Example configuration:

```yaml
model:
  checkpoint: openai-community/gpt2        # switch to openai-community/gpt2-medium for final runs
  checkpoint_local_path: null
  dtype: float32                            # CPU only; no BF16 support
  context_length: 1024

fine_tuning:
  method: lora                              # recommended for CPU; set to full for GPU runs
  lora:
    enabled: true
    rank: 8
    alpha: 16
    target_modules: [c_attn, c_proj]        # GPT-2 attention projection names

training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  num_train_epochs: 3
  warmup_ratio: 0.05
  lr_scheduler_type: cosine
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42
  logging_steps: 10
  eval_steps: 200
  save_steps: 500

output:
  output_dir: checkpoints/sft_run_001
  save_total_limit: 3
  push_to_hub: false
```

### 11.6.5 KG-Specific Architectural Extensions

The baseline fine-tuned model requires no architectural changes. KG extensions should be added incrementally in later phases.

Possible extensions:

- Entity classification head.
- Relation prediction head.
- Triple validity classifier.
- Graph consistency scoring head.
- Reward model head for KG-grounded RL.
- Retrieval-context adapter for graph neighborhoods.

These extensions must be configurable so that every experiment can compare:

- baseline checkpoint without KG fine-tuning,
- text-only KG integration,
- auxiliary objective integration,
- retrieval-augmented integration,
- and hybrid KG integration.

---

# 12. Reinforcement Learning

## 12.1 Objective

Use KG-derived reward signals to improve:

- factual correctness,
- reasoning consistency,
- answer qualification,
- refusal behavior,
- alignment with trusted graph evidence.

## 12.2 Composite Reward

    R_total =
      w_fact        * R_fact
    + w_consistency * R_consistency
    + w_reasoning   * R_reasoning
    + w_uncertainty * R_uncertainty
    + w_language    * R_language

Where:

- R_fact rewards graph-supported facts.
- R_consistency penalizes graph contradictions.
- R_reasoning rewards valid graph-path explanations.
- R_uncertainty rewards appropriate refusal when evidence is absent.
- R_language preserves fluency and helpfulness.

## 12.3 Reward Record

    {
      "prompt_id": "rl_000001",
      "prompt": "Explain how A influences C.",
      "kg_context": [
        "A regulates B",
        "B causes C"
      ],
      "model_output": "A influences C through B.",
      "reward": {
        "fact": 1.0,
        "consistency": 1.0,
        "reasoning": 0.8,
        "uncertainty": 0.0,
        "language": 0.9,
        "total": 0.92
      },
      "kg_snapshot": "kg_domain_v0.1"
    }

## 12.4 Reward Component Computation

Each component operates on the triple `(prompt, graph_context, model_output)` and returns a float in `[0.0, 1.0]`. Default weights: `w_fact=0.35`, `w_consistency=0.30`, `w_reasoning=0.15`, `w_uncertainty=0.10`, `w_language=0.10`. Weights must be configurable in `configs/reward.yaml`.

### R_fact — graph-supported facts

**Inputs:** `graph_context` (list of linearized hyperedges provided to the model), `model_output`.

**Algorithm:**

1. For each hyperedge in `graph_context`, check whether all head entity mentions, all tail entity mentions (canonical names or aliases, case-insensitive), and the relation keyword appear in `model_output` as substrings.
2. A hyperedge is considered "reflected" if at least one head entity, at least one tail entity, and the relation keyword are all present. Full credit requires all head and tail entities to be present.
3. `partial_score(edge) = (head_mentions_found / |head_ids|  +  tail_mentions_found / |tail_ids|) / 2`
4. `R_fact = mean(partial_score(edge) for edge in graph_context)` if `graph_context` is non-empty; else `R_fact = 1.0`.

### R_consistency — graph contradiction penalty

**Inputs:** `model_output`, `kg_snapshot` (validated relations only).

**Algorithm:**

1. Collect all `auto_validated` and `human_validated` canonical relations from the snapshot.
2. For each relation `(h, r, t)`: check whether `model_output` contains both entity names AND a negation word (`not`, `never`, `no`, `incorrect`, `false`, `wrong`) within a 10-word window.
3. `contradictions = count of relations matching the above pattern`.
4. `checked = count of relations where both entity names appear in model_output`.
5. `R_consistency = 1.0 - (contradictions / checked)` if `checked > 0`, clamped to `[0.0, 1.0]`; else `R_consistency = 1.0`.

### R_reasoning — valid graph-path explanation

**Inputs:** `model_output`, `kg_snapshot`, `task_type`.

**Algorithm (only applied when `task_type` is `multi_hop_qa` or `graph_path_explanation`; returns `1.0` for all other task types):**

1. Extract entity mentions from `model_output` by scanning for canonical names and aliases from the snapshot.
2. For each pair of mentioned entities `(A, C)`: search the snapshot for a two-hop path `A → B → C` where:
   - `A` appears in `head_ids` of a validated hyperedge `e1`, and
   - some entity `B` appears in `tail_ids` of `e1` AND in `head_ids` of a validated hyperedge `e2`, and
   - `C` appears in `tail_ids` of `e2`.
3. Check that `model_output` also mentions `B`.
4. `R_reasoning = 1.0` if a valid hyperedge path exists and `B` is mentioned; `0.5` if the path exists but `B` is absent; `0.0` if no valid path connects the mentioned entities.

### R_uncertainty — appropriate refusal

**Inputs:** `graph_context`, `model_output`, `task_type`.

**Algorithm (only applied when `task_type` is `missing_evidence_refusal` or `graph_context` is empty; returns `1.0` for all other cases):**

1. Check `model_output` for refusal indicators (case-insensitive substring match against): `cannot confirm`, `does not support`, `no evidence`, `insufficient information`, `cannot determine`, `not supported by`, `I don't have`, `unable to verify`.
2. `R_uncertainty = 1.0` if any indicator is found; `0.0` if the model makes a positive factual claim without qualification.

### R_language — output fluency proxy

**Inputs:** `model_output`, `expected_output` (the target response from the SFT record).

**Algorithm (word-count length ratio; replace with B1 perplexity in a later version):**

1. `output_words = len(model_output.split())`
2. `expected_words = max(len(expected_output.split()), 1)`
3. `ratio = output_words / expected_words`
4. `R_language = ratio / 0.5` if `ratio < 0.5`; `= 1.0` if `0.5 ≤ ratio ≤ 2.0`; `= 2.0 / ratio` if `ratio > 2.0`; clamped to `[0.0, 1.0]`.

---

# 13. C++ Implementation Requirements

## 13.1 Core Modules

Recommended module structure:

    src/
      core/
        ids.hpp
        errors.hpp
        logging.hpp
        hashing.hpp
        config.hpp

      corpus/
        document.hpp
        chunk.hpp
        corpus_reader.hpp
        corpus_manifest.hpp
        corpus_validator.hpp
        chunker.hpp

      prompts/
        prompt_template.hpp
        prompt_renderer.hpp
        prompt_manifest.hpp
        prompt_hash.hpp

      extraction/
        provider_interface.hpp
        openai_provider.hpp
        gemini_provider.hpp
        mock_provider.hpp
        replay_provider.hpp
        extraction_pipeline.hpp
        extraction_cache.hpp
        response_parser.hpp

      kg/
        entity.hpp
        relation.hpp
        schema.hpp
        graph_store.hpp
        graph_validator.hpp
        graph_snapshot.hpp
        graph_linearizer.hpp
        normalizer.hpp

      datasets/
        pretrain_builder.hpp
        sft_builder.hpp
        preference_builder.hpp
        eval_builder.hpp

      training/
        tokenizer.hpp
        dataloader.hpp
        finetune_config.hpp
        checkpoint_adapter.hpp

    scripts/
      finetune_sft.py
      finetune_dpo.py
      finetune_rl.py
      download_checkpoint.py
      evaluate_checkpoint.py

      evaluation/
        metrics.hpp
        kg_consistency.hpp
        hallucination.hpp
        reasoning_eval.hpp
        benchmark_runner.hpp

      rl/
        reward_function.hpp
        kg_reward.hpp
        preference_reward.hpp
        rl_dataset.hpp

      cli/
        main.cpp

## 13.2 Logging Specification

The `core/logging.hpp` module must implement the following contract.

**Log levels (in ascending severity):** `DEBUG`, `INFO`, `WARN`, `ERROR`.

**CLI flag behaviour:**

| Flag | Active levels |
|---|---|
| (default) | `INFO`, `WARN`, `ERROR` |
| `--verbose` | `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `--quiet` | `WARN`, `ERROR` only |

**Stderr output (human-readable):**

```
[INFO]  corpus_reader: loaded 1280 documents from data/raw
[WARN]  corpus_validator: 3 documents missing author metadata
[ERROR] corpus_validator: duplicate document ID doc_000042 — aborting
```

Format: `[LEVEL]  module_name: message`

**Structured log file (`--log-file path/to/log.jsonl`):**

Each log event is one JSON object per line:

```json
{"timestamp": "2026-05-12T00:00:01Z", "level": "WARN", "module": "corpus_validator", "message": "3 documents missing author metadata", "context": {"count": 3}}
```

The `context` field is optional and may carry structured key-value pairs relevant to the event (file paths, counts, hashes, IDs).

**Hard failure behaviour:** `ERROR`-level log entries must always precede a non-zero process exit. A process must not exit non-zero without first emitting an `ERROR` log entry explaining the reason.

**Corpus validation hard failures** (Section 5.7) log at `ERROR` and exit with code `1`. Corpus warnings log at `WARN` and do not affect exit code.

## 13.3 Storage Formats

Minimum required formats:

- JSONL for documents, chunks, entities, relations, datasets.
- YAML for configuration and manifests.
- JSON for validation reports and benchmark results.
- CSV for optional human-readable summaries.

Optional later formats:

- SQLite,
- DuckDB,
- Parquet,
- Kùzu,
- RDF/Turtle.

---

# 14. CLI Specification

Executable name:

    slmkg

Global command shape:

    slmkg <command> <subcommand> [options]

Global options:

    --config path/to/config.yaml
    --verbose
    --quiet
    --dry-run
    --seed 42
    --output path
    --log-file path/to/log.jsonl

## 14.1 Project Initialization

    slmkg init --project my_slmkg_project

The `init` command creates the full directory structure and writes starter config files. It does not download checkpoints or touch external APIs.

Directories created:

    my_slmkg_project/
      configs/          ← starter corpus.yaml, kg_extraction.yaml, model.yaml, training.yaml
      prompts/          ← placeholder system and user prompt files
      data/raw/documents/
      data/processed/
      data/kg/
      data/datasets/pretrain/ sft/ preference/ eval/
      checkpoints/base/
      reports/
      benchmarks/
      tests/fixtures/mock_provider/
      tests/fixtures/golden/
      tests/fixtures/replay/
      scripts/          ← symlink or copy of the project scripts/ directory
      .cache/kg_extraction/

All starter config files contain the full schema with commented-out optional fields. They are functional with default values and require only the corpus path and API key environment variable to be filled in before running Phase 1.

## 14.2 Corpus Commands

Ingest corpus:

    slmkg corpus ingest \
      --input data/raw \
      --output data/processed/corpus_v0.1 \
      --config configs/corpus.yaml

Validate corpus:

    slmkg corpus validate \
      --corpus data/processed/corpus_v0.1

Corpus statistics:

    slmkg corpus stats \
      --corpus data/processed/corpus_v0.1 \
      --output reports/corpus_stats.json

## 14.3 Prompt Commands

Validate prompt templates:

    slmkg prompt validate \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt \
      --schema configs/kg_extraction_output_schema.json

Render prompt for inspection:

    slmkg prompt render \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt \
      --chunk-id chunk_000001 \
      --corpus data/processed/corpus_v0.1 \
      --schema kg/schema.yaml \
      --output reports/rendered_prompt_preview.txt

Hash prompt set:

    slmkg prompt hash \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt

## 14.4 KG Commands

Validate schema:

    slmkg kg schema-validate \
      --schema kg/schema.yaml

Extract KG from corpus:

    slmkg kg extract \
      --corpus data/processed/corpus_v0.1 \
      --schema kg/schema.yaml \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt \
      --output data/kg/kg_v0.1 \
      --config configs/kg_extraction.yaml

Extract KG using replay mode:

    slmkg kg extract \
      --corpus data/processed/corpus_v0.1 \
      --schema kg/schema.yaml \
      --system-prompt prompts/kg_system_prompt_v0.1.txt \
      --user-prompt prompts/kg_user_prompt_v0.1.txt \
      --output data/kg/kg_v0.1_replay \
      --config configs/kg_extraction.yaml \
      --replay

Normalize KG:

    slmkg kg normalize \
      --input data/kg/kg_v0.1 \
      --output data/kg/kg_v0.1_normalized

Validate KG:

    slmkg kg validate \
      --kg data/kg/kg_v0.1_normalized \
      --schema kg/schema.yaml \
      --output reports/kg_validation_v0.1.json

Create KG snapshot:

    slmkg kg snapshot \
      --kg data/kg/kg_v0.1_normalized \
      --snapshot-id kg_domain_v0.1 \
      --output data/kg/snapshots/kg_domain_v0.1

## 14.5 Dataset Commands

Build pre-training dataset:

    slmkg dataset build-pretrain \
      --corpus data/processed/corpus_v0.1 \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output data/datasets/pretrain/pretrain_v0.1.jsonl \
      --config configs/pretrain_dataset.yaml

Build SFT dataset:

    slmkg dataset build-sft \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output data/datasets/sft/sft_v0.1.jsonl \
      --config configs/sft_dataset.yaml

Build preference dataset:

    slmkg dataset build-preference \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output data/datasets/preference/pref_v0.1.jsonl \
      --config configs/preference_dataset.yaml

Build evaluation dataset:

    slmkg dataset build-eval \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output data/datasets/eval/eval_v0.1.jsonl \
      --config configs/eval_dataset.yaml

Audit train/validation/test/benchmark splits for leakage:

    slmkg dataset audit-splits \
      --train data/datasets/sft/sft_v0.1_train.jsonl \
      --validation data/datasets/sft/sft_v0.1_val.jsonl \
      --test data/datasets/sft/sft_v0.1_test.jsonl \
      --benchmark data/datasets/eval/eval_v0.1.jsonl \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output reports/split_audit_v0.1.json

The audit command checks for:

- duplicate record IDs across splits,
- identical `input` strings across train and test,
- shared `evidence_relation_ids` between train and benchmark,
- shared entity canonical names between train and benchmark held-out sets,
- shared graph paths between train and benchmark multi-hop records.

The output JSON reports leakage counts per category and lists the offending record IDs. The command exits with a non-zero code if any benchmark leakage is detected. Validation split leakage produces a warning only.

## 14.6 Training Commands

Download pre-trained checkpoint:

    slmkg train download \
      --checkpoint openai-community/gpt2 \
      --output checkpoints/base/gpt2-small

    slmkg train download \
      --checkpoint openai-community/gpt2-medium \
      --output checkpoints/base/gpt2-medium

Supervised fine-tune on KG-derived dataset:

    slmkg train sft \
      --base-checkpoint checkpoints/base/gpt2-small \
      --dataset data/datasets/sft/sft_v0.1.jsonl \
      --model-config configs/model.yaml \
      --training-config configs/sft.yaml \
      --output checkpoints/sft_run_001

Fine-tune on natural text only (for baseline B1):

    slmkg train sft \
      --base-checkpoint checkpoints/base/gpt2-small \
      --dataset data/datasets/pretrain/pretrain_v0.1.jsonl \
      --model-config configs/model.yaml \
      --training-config configs/pretrain.yaml \
      --output checkpoints/baseline_run_001

Preference optimization (DPO):

    slmkg train dpo \
      --base-checkpoint checkpoints/sft_run_001 \
      --preference-dataset data/datasets/preference/pref_v0.1.jsonl \
      --training-config configs/dpo.yaml \
      --output checkpoints/dpo_run_001

Reinforcement learning with KG reward:

    slmkg train rl \
      --base-checkpoint checkpoints/sft_run_001 \
      --preference-dataset data/datasets/preference/pref_v0.1.jsonl \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --reward-config configs/reward.yaml \
      --output checkpoints/rl_run_001

The `slmkg train` commands are thin CLI wrappers that invoke the corresponding Python scripts in `scripts/` and write a training manifest on completion. No training logic is implemented in C++.

## 14.7 Evaluation Commands

Run evaluation:

    slmkg eval run \
      --checkpoint checkpoints/sft_run_001 \
      --benchmark benchmarks/kg_consistency_v0.1.yaml \
      --kg data/kg/snapshots/kg_domain_v0.1 \
      --output reports/eval_sft_run_001.json

Compare benchmark runs:

    slmkg benchmark compare \
      --runs reports/eval_baseline.json reports/eval_kg_sft.json reports/eval_kg_rl.json \
      --output reports/benchmark_comparison.md

---

# 15. Unit Testing Strategy

## 15.1 Test Categories

The project should include:

- unit tests,
- integration tests,
- golden-file tests,
- provider replay tests,
- prompt rendering tests,
- schema validation tests,
- fuzz tests,
- regression tests,
- benchmark smoke tests.

## 15.2 Corpus Tests

Required tests:

    CorpusReaderTest.LoadsValidJsonl
    CorpusReaderTest.RejectsMalformedJsonl
    CorpusValidatorTest.RejectsDuplicateDocumentIds
    ChunkerTest.DoesNotCreateEmptyChunks
    ChunkerTest.RespectsMaxChunkSize
    CorpusSplitTest.IsDeterministicWithSeed
    CorpusManifestTest.HashStableForSameInput

## 15.3 Prompt Tests

Required tests:

    PromptRendererTest.SubstitutesAllVariables
    PromptRendererTest.FailsOnMissingVariable
    PromptRendererTest.PreservesChunkText
    PromptRendererTest.RendersSchemaCorrectly
    PromptHashTest.HashStableForSamePrompt
    PromptHashTest.HashChangesWhenPromptChanges
    PromptManifestTest.RecordsPromptPathsAndHashes

## 15.4 Extraction Provider Tests

Required tests:

    MockProviderTest.ReturnsDeterministicResponse
    ReplayProviderTest.ReplaysRecordedResponse
    ProviderConfigTest.LoadsOpenAIConfig
    ProviderConfigTest.LoadsGeminiConfig
    ProviderConfigTest.DoesNotExposeApiKeyInLogs
    ResponseParserTest.RejectsInvalidJson
    ResponseParserTest.RejectsSchemaViolation

## 15.5 KG Tests

Required tests:

    KgSchemaTest.LoadsValidSchema
    KgSchemaTest.RejectsUnknownEntityType
    KgSchemaTest.RejectsUnknownRelationType
    KgSchemaTest.ValidatesDomainRangeConstraints
    GraphStoreTest.RoundTripsEntities
    GraphStoreTest.RoundTripsRelations
    EntityNormalizerTest.MergesAliases
    RelationNormalizerTest.DeduplicatesRelations
    GraphSnapshotTest.HashStableUnderSortedInput
    GraphSnapshotTest.HashChangesWhenFactChanges

## 15.6 Dataset Tests

Required tests:

    GraphLinearizerTest.LinearizesTriple
    GraphLinearizerTest.LinearizesPath
    NegativeSamplerTest.DoesNotGenerateTrueFactAsNegative
    SftBuilderTest.GeneratesSingleHopQa
    SftBuilderTest.GeneratesMissingEvidenceRefusal
    PreferenceBuilderTest.GeneratesChosenRejectedPairs
    EvalBuilderTest.HoldsOutTestFacts

## 15.7 Reward Tests

Required tests:

    KgRewardTest.RewardsSupportedClaim
    KgRewardTest.PenalizesContradiction
    KgRewardTest.RewardsCorrectRefusal
    KgRewardTest.PenalizesUnsupportedClaim
    CompositeRewardTest.WeightedSumIsCorrect

## 15.8 Integration Tests

Required end-to-end fixture tests:

    PipelineTest.CorpusToKgExtractionWithMockProvider
    PipelineTest.CorpusToKgExtractionWithReplayProvider
    PipelineTest.KgToPretrainDataset
    PipelineTest.KgToSftDataset
    PipelineTest.KgToEvalDataset
    PipelineTest.EndToEndTinyBenchmark

These tests must run without live external API calls.

## 15.9 Golden-File Tests

Golden-file tests should be used for:

- rendered prompts,
- graph linearization,
- dataset generation,
- validation reports,
- benchmark reports,
- manifest generation.

Golden files should be updated only through explicit developer action.

---

# 16. Benchmarking Specification

## 16.1 Baseline Models

Required baselines:

    B0: pre-trained checkpoint with no further fine-tuning (frozen baseline)
    B1: pre-trained checkpoint fine-tuned on natural text only (no KG data)
    B2: pre-trained checkpoint fine-tuned with graph-linearized data
    B3: pre-trained checkpoint fine-tuned with graph-verbalized data
    B4: pre-trained checkpoint with graph-grounded SFT
    B5: pre-trained checkpoint with KG-based RL
    B6: optional retrieval-augmented KG model
    B7: optional graph-embedding model

B0 is the zero-shot performance of the pre-trained checkpoint on the benchmark suite, before any fine-tuning. B1 is the fine-tuned baseline with no KG involvement. B2 through B5 add KG-derived data and signal incrementally. Every benchmark report must state which pre-trained checkpoint was used (GPT-2 Small for development runs, GPT-2 Medium for final experiments).

## 16.2 Benchmark Categories

The benchmark suite should measure:

| Category | Example Metrics |
|---|---|
| General language quality | perplexity, validation loss |
| Factual accuracy | graph-supported QA accuracy |
| KG consistency | contradiction rate, unsupported claim rate |
| Entity grounding | entity linking accuracy, disambiguation accuracy |
| Relation understanding | relation classification accuracy |
| Multi-hop reasoning | path QA accuracy, explanation validity |
| Hallucination | unsupported factual claim rate |
| Refusal behavior | correct refusal rate when KG lacks evidence |
| Robustness | paraphrase and adversarial prompt performance |
| Graph update behavior | accuracy after graph changes |
| Efficiency | training time, memory usage, tokens per second |
| Data efficiency | performance by training-token budget |

## 16.2a Metric Computation Definitions

All metrics are computed by `scripts/evaluate_checkpoint.py` on the held-out test split. The normalization function used throughout is:

```
normalize(s) = s.lower().strip().translate(str.maketrans('', '', string.punctuation)).split()
             → rejoined with single spaces
```

| Metric | Computation |
|---|---|
| **Perplexity** | `exp(mean cross-entropy loss)` over all tokens in the held-out natural-text test split. Computed with `model.eval()`, no sampling. |
| **Graph-supported QA accuracy** | For `task_type` in `[single_hop_qa, multi_hop_qa]`: `exact_match(normalize(expected_output), normalize(model_output))`. Report as fraction over all QA eval records. |
| **Contradiction detection accuracy** | For `task_type == contradiction_detection`: score 1.0 if the model output contains the word `contradiction`, `contradicts`, `incorrect`, `false`, or `conflicts`; else 0.0. |
| **Unsupported claim rate** | Fraction of eval records with non-empty `graph_context` where `R_fact < 0.5`. |
| **Correct refusal rate** | For `task_type == missing_evidence_refusal`: fraction of records where `R_uncertainty == 1.0`. |
| **Entity linking accuracy** | For `task_type == entity_disambiguation`: `exact_match(normalize(expected_entity_name), normalize(model_output))` where `expected_entity_name` is the canonical name of the target entity. |
| **Relation classification accuracy** | For `task_type == relation_explanation`: check whether the expected `relation_type` string or any schema-defined synonym appears in `normalize(model_output)`. |
| **Multi-hop path accuracy** | For `task_type in [multi_hop_qa, graph_path_explanation]`: mean `R_reasoning` score across all multi-hop eval records. |
| **Explanation validity** | Fraction of `graph_path_explanation` records where `R_reasoning == 1.0`. |

Metrics that do not apply to a given checkpoint or dataset split are reported as `null`, not zero.

## 16.3 Benchmark Datasets

Required benchmark sets:

- Graph QA benchmark.
- Multi-hop reasoning benchmark.
- Contradiction benchmark.
- Missing evidence benchmark.
- Entity disambiguation benchmark.
- Relation understanding benchmark.
- Graph update benchmark.
- General language benchmark.

## 16.4 Ablation Studies

Recommended ablations:

    A1: no KG data
    A2: KG triples only
    A3: KG verbalized text only
    A4: KG path data only
    A5: KG contradiction data only
    A6: KG SFT without KG pre-training
    A7: KG pre-training without KG SFT
    A8: KG SFT without RL
    A9: KG SFT plus KG reward
    A10: retrieval at inference versus distilled KG only
    A11: different system prompts for KG extraction
    A12: different user prompts for KG extraction
    A13: OpenAI extraction versus Gemini extraction
    A14: human-reviewed KG versus fully automatic KG

## 16.5 Prompt-Aware Benchmarking

Because the KG is extracted through user-defined prompts, benchmark reports must record:

- system prompt hash,
- user prompt hash,
- prompt set ID,
- provider,
- model,
- extraction temperature,
- extraction timestamp,
- schema version,
- KG snapshot ID.

Prompt changes should be treated as experimental variables.

## 16.6 Benchmark Report Format

    benchmark_run:
      id: benchmark_2026_05_12_001
      checkpoint: checkpoints/sft_run_001
      kg_snapshot: kg_domain_v0.1
      corpus_version: corpus_v0.1
      prompt_set_id: kg_extract_promptset_v0.1
      system_prompt_hash: sha256:...
      user_prompt_hash: sha256:...
      provider: openai
      provider_model: gpt-4.1-mini
      benchmark_version: kg_benchmark_v0.1
      metrics:
        graph_qa_accuracy: 0.74
        multi_hop_accuracy: 0.51
        contradiction_detection_accuracy: 0.81
        unsupported_claim_rate: 0.18
        correct_refusal_rate: 0.67
        perplexity: 23.4
      environment:
        seed: 42
        compiler: clang
        build_type: release

---

# 17. Evaluation and Failure Analysis

## 17.1 Evaluation Categories

The project should evaluate:

- general language quality,
- factual accuracy,
- KG consistency,
- entity disambiguation,
- relation understanding,
- multi-hop reasoning,
- hallucination rate,
- refusal quality,
- robustness to paraphrase,
- robustness to contradiction,
- graph update behavior,
- prompt sensitivity.

## 17.2 Failure Categories

Failure records should use standard labels:

    unsupported_claim
    wrong_entity
    wrong_relation
    incorrect_path
    overconfident_answer
    unnecessary_refusal
    contradiction
    fluent_but_false
    format_error
    prompt_extraction_error
    kg_validation_error
    evidence_mismatch

Example failure record:

    {
      "id": "fail_000123",
      "prompt": "How is A related to C?",
      "expected": "A regulates B, and B causes C.",
      "actual": "A directly causes C.",
      "failure_type": "incorrect_path",
      "kg_evidence": ["rel_001", "rel_002"],
      "notes": "Model collapsed a two-hop path into an unsupported direct relation."
    }

## 17.3 Evaluation Record Schema

Every evaluation dataset record must conform to the following schema. Records are stored in JSONL format.

```json
{
  "id": "eval_000001",
  "task_type": "single_hop_qa",
  "split": "test",
  "instruction": "Answer using only the provided graph evidence.",
  "input": "What is the capital of France?",
  "graph_context": ["Paris capital_of France"],
  "expected_output": "Paris",
  "evaluation_method": "exact_match_normalized",
  "evidence_relation_ids": ["rel_000042"],
  "kg_snapshot": "kg_domain_v0.1",
  "difficulty": "easy"
}
```

Required fields: `id`, `task_type`, `split`, `instruction`, `input`, `expected_output`, `evaluation_method`, `kg_snapshot`.

Optional fields: `graph_context` (defaults to empty list), `evidence_relation_ids`, `difficulty`.

Supported `evaluation_method` values:

| Value | Description |
|---|---|
| `exact_match_normalized` | `normalize(expected_output) == normalize(model_output)` |
| `substring_match` | `normalize(expected_output)` appears as a substring in `normalize(model_output)` |
| `entity_match` | At least one entity from `evidence_relation_ids` head or tail appears in `normalize(model_output)` |
| `refusal_match` | `model_output` contains a refusal indicator; used for `missing_evidence_refusal` task type |
| `path_match` | `R_reasoning` algorithm; used for `multi_hop_qa` and `graph_path_explanation` |

## 17.4 Inference Settings

The evaluation script must use the following fixed settings to ensure reproducibility across runs and across baselines.

```yaml
inference:
  decoding: greedy          # do_sample: false; mandatory for benchmark evaluation
  temperature: 1.0          # ignored when do_sample is false; stated for clarity
  top_k: 0
  top_p: 1.0
  max_new_tokens: 256
  pad_token_id: eos_token_id  # GPT-2 has no dedicated pad token; use EOS
  seed: 42                  # set torch.manual_seed and random.seed before any inference
```

Greedy decoding is mandatory for all benchmark runs. Sampling must not be used during evaluation because it introduces non-determinism that makes metric comparisons invalid across runs.

## 17.5 Evaluation Output Schema

The evaluation script writes one JSON file per run.

```json
{
  "eval_run_id": "eval_run_001",
  "created_at": "2026-05-12T00:00:00Z",
  "checkpoint": "checkpoints/b4_kg_sft_small",
  "base_model": "openai-community/gpt2",
  "kg_snapshot": "kg_domain_v0.1",
  "benchmark_version": "kg_benchmark_v0.1",
  "decoding": {
    "method": "greedy",
    "max_new_tokens": 256
  },
  "seed": 42,
  "metrics": {
    "graph_qa_accuracy": 0.74,
    "multi_hop_accuracy": 0.51,
    "contradiction_detection_accuracy": 0.81,
    "unsupported_claim_rate": 0.18,
    "correct_refusal_rate": 0.67,
    "entity_linking_accuracy": 0.69,
    "relation_classification_accuracy": 0.72,
    "perplexity": 23.4
  },
  "per_record_results": [
    {
      "id": "eval_000001",
      "task_type": "single_hop_qa",
      "model_output": "Paris is the capital of France.",
      "score": 1.0,
      "evaluation_method": "exact_match_normalized",
      "failure_type": null
    }
  ]
}
```

The `failure_type` field must use the vocabulary defined in Section 17.2. It is `null` when `score == 1.0`.

---

# 18. Training Run Manifest

Every training run must generate a manifest.

    training_run:
      id: sft_run_001
      created_at: 2026-05-12T00:00:00Z
      base_checkpoint: pretrain_run_001
      output_checkpoint: checkpoints/sft_run_001
      model_config: configs/model.yaml
      training_config: configs/sft.yaml
      dataset:
        id: sft_v0.1
        path: data/datasets/sft/sft_v0.1.jsonl
        hash: sha256:...
      kg_snapshot: kg_domain_v0.1
      corpus_version: corpus_v0.1
      prompt_set_id: kg_extract_promptset_v0.1
      system_prompt_hash: sha256:...
      user_prompt_hash: sha256:...
      seed: 42
      metrics:
        train_loss_final: 1.92
        validation_loss_final: 2.11

---

# 19. Risk Register

| Risk | Mitigation |
|---|---|
| KG contains incorrect facts | Track provenance, confidence, evidence, validation status, and source trust |
| User prompt causes over-extraction | Validate against schema and require evidence spans |
| User prompt causes under-extraction | Compare extraction density across prompt versions |
| Prompt changes break reproducibility | Store prompt hashes and prompt set IDs in all manifests |
| External LLM returns invalid JSON | Use strict schema validation, retries, and rejected candidate logs |
| API cost grows unexpectedly | Use caching, replay mode, dry-run mode, chunk limits, and concurrency limits |
| Model memorizes graph triples | Use held-out entities, held-out relations, held-out paths, and paraphrased tests |
| Synthetic graph data hurts fluency | Maintain natural text mixture and monitor perplexity |
| Reward model overfits narrow graph checks | Use mixed rewards and held-out adversarial benchmarks |
| Train-test leakage through graph facts | Split by document, entity, relation, and path |
| C++ slows experimentation | Training uses Python (HuggingFace); C++ is scoped to the pipeline only |
| Benchmarks become too easy | Include contradiction, missing evidence, paraphrase, and multi-hop tests |

---

# 20. Recommended Development Roadmap

## Phase 0: Specification and Fixtures

**Stack:** Text editor, YAML, JSON. No build step.

Deliverables:

- final technical specification,
- tiny test corpus,
- tiny KG schema,
- prompt templates,
- mock provider outputs,
- expected golden files.

Acceptance criteria:

- fixture corpus can be loaded,
- prompt renderer works,
- mock provider returns deterministic extraction results.

## Phase 1: Corpus and Prompt Infrastructure

**Stack:** C++17, CMake 3.20+, nlohmann/json, yaml-cpp, OpenSSL, Catch2.

Deliverables:

- corpus ingestion,
- chunking,
- corpus validation,
- prompt renderer,
- prompt hashing,
- prompt manifest generation.

Acceptance criteria:

- corpus manifest generated,
- prompt hashes generated,
- prompt rendering tests pass.

## Phase 2: KG Extraction Layer

**Stack:** C++17, CMake 3.20+, nlohmann/json, yaml-cpp, libcurl, OpenSSL, Catch2. External: OpenAI or Gemini API (live extraction only; mock and replay providers need no API keys).

Deliverables:

- provider interface,
- mock provider,
- replay provider,
- OpenAI provider,
- Gemini provider,
- response parser,
- output schema validator,
- extraction cache.

Acceptance criteria:

- KG extraction works with mock provider,
- replay mode works,
- invalid outputs are rejected,
- no live API calls are needed for CI.

## Phase 3: KG Normalization and Validation

**Stack:** C++17, CMake 3.20+, nlohmann/json, yaml-cpp, Catch2. Fully offline.

Deliverables:

- entity normalizer,
- alias resolver,
- relation validator,
- graph snapshot generator,
- validation report.

Acceptance criteria:

- duplicate entities merge correctly,
- invalid relations are rejected,
- KG snapshot manifest is generated.

## Phase 4: Dataset Generation

**Stack:** C++17, CMake 3.20+, nlohmann/json, Catch2. Fully offline.

Deliverables:

- pre-training dataset builder,
- SFT dataset builder,
- preference dataset builder,
- evaluation dataset builder,
- negative sample generator.

Acceptance criteria:

- graph-derived datasets are generated,
- true facts are not used as negative samples,
- dataset manifests reference KG snapshots.

## Phase 5: Baseline SLM

**Stack:** Python 3.10+, torch (CPU build), transformers, peft, trl, accelerate, datasets. External: one-time GPT-2 checkpoint download from HuggingFace. No GPU, no compilation.

Deliverables:

- downloaded and validated pre-trained checkpoints (GPT-2 Small for development, GPT-2 Medium for final runs),
- Python fine-tuning scripts (`scripts/finetune_sft.py`) integrated with C++ JSONL outputs,
- `slmkg train download` and `slmkg train sft` CLI commands operational,
- B0 zero-shot benchmark report (frozen checkpoint),
- B1 natural-text fine-tuning run with checkpoint manifest,
- B1 benchmark report establishing the KG-free fine-tuning baseline.

Acceptance criteria:

- pre-trained checkpoint loads and generates coherent text,
- fine-tuning on natural text completes without numerical instability,
- validation loss is reported and logged in the training manifest,
- benchmark runner produces comparable B0 and B1 reports,
- no KG data is present in the B1 training set.

## Phase 6: KG-Augmented SLM

**Stack:** Same Python virtual environment as Phase 5. No additional installs. Datasets produced by the C++ pipeline (Phase 4).

Deliverables:

- graph-text pre-training mixture,
- graph-grounded SFT,
- KG consistency benchmark,
- ablation reports.

Acceptance criteria:

- KG-enhanced model improves graph QA or consistency metrics,
- fluency does not degrade beyond accepted threshold.

## Phase 7: KG-Based RL

**Stack:** Same Python virtual environment as Phase 5. trl DPO/PPO trainers. No additional installs.

Deliverables:

- KG reward functions,
- preference data,
- RL training prototype,
- reward-hacking analysis.

Acceptance criteria:

- KG reward improves held-out factuality or refusal metrics,
- failures are categorized,
- reward overfitting is tested.

---

# 21. Open Research Questions

The project should investigate:

1. Which KG integration method works best for compact models?
2. Does graph-derived pre-training improve factuality or merely increase memorization?
3. How sensitive is KG quality to the user-defined system prompt?
4. How sensitive is KG quality to the user-defined user prompt?
5. Does provider choice materially affect KG quality?
6. Can graph-grounded SFT improve refusal behavior?
7. Can KG-based rewards improve reasoning consistency?
8. How much synthetic graph data is useful before fluency degrades?
9. Should the KG be distilled into the model, kept external, or both?
10. Can small models learn reusable reasoning patterns from graph paths?
11. How should graph updates be reflected in training, fine-tuning, or inference?

---

# 22. Summary

This specification defines a C++-first pipeline for fine-tuning a small language model with knowledge graphs as first-class design components.

The pipeline uses C++ for all stages from corpus ingestion through dataset generation. Fine-tuning and reinforcement learning are implemented in Python using HuggingFace `transformers`, `trl`, and `peft`, invoked through the `slmkg train` CLI commands. The baseline model is a pre-trained GPT-2 checkpoint — GPT-2 Small (124M) for development and ablations, GPT-2 Medium (355M) for final experiments — rather than a model trained from scratch. Both run on CPU without a GPU requirement.

The key refinement is that the KG is explicitly extracted from the input document corpus using external LLM API calls. The user controls the system prompt and user prompt, and these prompts become part of the reproducible experimental record.

The resulting architecture provides:

- corpus ingestion,
- prompt-configurable KG extraction,
- provider abstraction,
- replayable API calls,
- strict output validation,
- provenance-rich graph construction,
- versioned KG snapshots,
- graph-derived training datasets,
- KG-grounded supervised fine-tuning of a pre-trained checkpoint,
- KG-based reinforcement learning,
- prompt-aware benchmarking,
- and a robust unit-testing and CLI strategy.

The most important implementation rule is:

> No KG fact should enter the training or evaluation pipeline unless it can be traced back to a corpus document, extraction prompt, provider response, schema validation step, and KG snapshot.

That traceability is the difference between a useful neuro-symbolic SLM experiment and a glittering bag of hallucinated graph confetti.

---

# 23. Developer Instructions

This section provides environment setup, build commands, and verification steps for each phase. All C++ phases share a common CMake project. Python fine-tuning phases use a dedicated virtual environment that is set up once and reused across Phases 5, 6, and 7.

---

## 23.1 Prerequisites

Install the following before starting any phase.

**C++ toolchain (required for Phases 1–4):**

- clang++ 14+ or g++ 12+ with C++17 support
- CMake 3.20+
- OpenSSL development headers (`libssl-dev` on Debian/Ubuntu; `openssl` via Homebrew on macOS)
- libcurl development headers (`libcurl4-openssl-dev` on Debian/Ubuntu; `curl` via Homebrew on macOS)
- Git

**Python toolchain (required for Phases 5–7):**

- Python 3.10+
- pip

**Network access is required only for:**

- OpenAI or Gemini API calls during KG extraction (Phase 2 and later, live mode only)
- One-time GPT-2 checkpoint download from HuggingFace (Phase 5)

All other steps run fully offline once the checkpoints are cached locally.

---

## 23.2 C++ Build Setup (Phases 1–4)

All C++ phases share the same CMake project. Third-party C++ libraries are fetched automatically using CMake FetchContent. System libraries (libcurl, OpenSSL) must be installed separately via the system package manager.

**Recommended `CMakeLists.txt` dependency block:**

```cmake
include(FetchContent)

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
  DOWNLOAD_NO_EXTRACT TRUE)

FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0)

FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4)

FetchContent_MakeAvailable(nlohmann_json yaml-cpp Catch2)
```

**Debug build and full test run:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Release build:**

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
```

The `slmkg` executable is produced at `build/slmkg` (or `build-release/slmkg`).

---

## 23.3 Phase 0: Specification and Fixtures

**Language:** None. No build step required.

Create the project directory structure:

```bash
mkdir -p data/raw/documents data/processed data/kg data/datasets
mkdir -p prompts configs checkpoints reports benchmarks
mkdir -p tests/fixtures/golden src scripts
```

Create a tiny fixture corpus of 3–5 short `.txt` or `.md` files in `data/raw/documents/`. Create a minimal `configs/kg_schema.yaml` with 2–3 entity types and 2–3 relation types. Write one fixture mock provider response at `tests/fixtures/mock_extraction_response.json` matching the output schema in Section 8.

Write expected golden files for prompt rendering and graph linearization to `tests/fixtures/golden/`. These files are the reference outputs for golden-file tests in later phases.

**Verify:** Validate all YAML and JSON fixture files:

```bash
python3 -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('configs/*.yaml')]"
python3 -c "import json, glob; [json.load(open(f)) for f in glob.glob('tests/fixtures/*.json')]"
```

---

## 23.4 Phase 1: Corpus and Prompt Infrastructure

**Language:** C++17  
**Build system:** CMake 3.20+  
**Core libraries:** nlohmann/json (FetchContent), yaml-cpp (FetchContent), OpenSSL (system, SHA256)  
**Testing:** Catch2 (FetchContent)

**Build and run corpus and prompt tests:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build \
  -R "CorpusReader|Chunker|PromptRenderer|PromptHash|CorpusManifest" \
  --output-on-failure
```

**Verify with fixture corpus:**

```bash
./build/slmkg corpus ingest \
  --input data/raw \
  --output data/processed/corpus_v0.1 \
  --config configs/corpus.yaml

./build/slmkg corpus validate \
  --corpus data/processed/corpus_v0.1

./build/slmkg prompt render \
  --system-prompt prompts/kg_system_prompt_v0.1.txt \
  --user-prompt prompts/kg_user_prompt_v0.1.txt \
  --chunk-id chunk_000001 \
  --corpus data/processed/corpus_v0.1 \
  --schema configs/kg_schema.yaml \
  --output reports/rendered_prompt_preview.txt
```

Expected output: `corpus_manifest.yaml` and `chunks.jsonl` written to `data/processed/corpus_v0.1/`, and a rendered prompt file with all `{{template_variables}}` substituted.

---

## 23.5 Phase 2: KG Extraction Layer

**Language:** C++17  
**Build system:** CMake 3.20+  
**Core libraries:** nlohmann/json, yaml-cpp, libcurl (system), OpenSSL (system)  
**Testing:** Catch2; mock and replay providers must pass all tests without live API calls

**Additional CMake linkage required:**

```cmake
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
target_link_libraries(slmkg PRIVATE CURL::libcurl OpenSSL::SSL OpenSSL::Crypto)
```

**Build and test (no API keys needed):**

```bash
cmake --build build --parallel
ctest --test-dir build \
  -R "MockProvider|ReplayProvider|ResponseParser|ProviderConfig" \
  --output-on-failure
```

**Verify with mock provider (offline):**

```bash
./build/slmkg kg extract \
  --corpus data/processed/corpus_v0.1 \
  --schema configs/kg_schema.yaml \
  --system-prompt prompts/kg_system_prompt_v0.1.txt \
  --user-prompt prompts/kg_user_prompt_v0.1.txt \
  --output data/kg/kg_v0.1 \
  --config configs/kg_extraction.yaml \
  --mock
```

**Live extraction with OpenAI (requires API key):**

```bash
export OPENAI_API_KEY=sk-...
./build/slmkg kg extract \
  --corpus data/processed/corpus_v0.1 \
  --schema configs/kg_schema.yaml \
  --system-prompt prompts/kg_system_prompt_v0.1.txt \
  --user-prompt prompts/kg_user_prompt_v0.1.txt \
  --output data/kg/kg_v0.1 \
  --config configs/kg_extraction.yaml
```

API responses are cached in `.cache/kg_extraction/`. Re-running the same chunks reuses the cache and makes no further API calls.

---

## 23.6 Phase 3: KG Normalization and Validation

**Language:** C++17  
**Build system:** CMake 3.20+  
**Core libraries:** nlohmann/json, yaml-cpp  
**Testing:** Catch2. Fully offline.

**Build and test:**

```bash
cmake --build build --parallel
ctest --test-dir build \
  -R "KgSchema|GraphStore|EntityNormalizer|RelationNormalizer|GraphSnapshot" \
  --output-on-failure
```

**Verify:**

```bash
./build/slmkg kg normalize \
  --input data/kg/kg_v0.1 \
  --output data/kg/kg_v0.1_normalized

./build/slmkg kg validate \
  --kg data/kg/kg_v0.1_normalized \
  --schema configs/kg_schema.yaml \
  --output reports/kg_validation_v0.1.json

./build/slmkg kg snapshot \
  --kg data/kg/kg_v0.1_normalized \
  --snapshot-id kg_domain_v0.1 \
  --output data/kg/snapshots/kg_domain_v0.1
```

Expected output: `reports/kg_validation_v0.1.json` with zero hard errors, and a `manifest.yaml` in `data/kg/snapshots/kg_domain_v0.1/`.

---

## 23.7 Phase 4: Dataset Generation

**Language:** C++17  
**Build system:** CMake 3.20+  
**Core libraries:** nlohmann/json  
**Testing:** Catch2, golden-file tests. Fully offline.

**Build and test:**

```bash
cmake --build build --parallel
ctest --test-dir build \
  -R "GraphLinearizer|NegativeSampler|SftBuilder|PreferenceBuilder|EvalBuilder" \
  --output-on-failure
```

**Verify:**

```bash
./build/slmkg dataset build-sft \
  --kg data/kg/snapshots/kg_domain_v0.1 \
  --output data/datasets/sft/sft_v0.1.jsonl \
  --config configs/sft_dataset.yaml

./build/slmkg dataset build-eval \
  --kg data/kg/snapshots/kg_domain_v0.1 \
  --output data/datasets/eval/eval_v0.1.jsonl \
  --config configs/eval_dataset.yaml
```

Inspect the first few generated records to confirm the expected structure:

```bash
head -n 3 data/datasets/sft/sft_v0.1.jsonl | python3 -m json.tool
```

---

## 23.8 Phase 5: Baseline SLM

**Language:** Python 3.10+  
**Core libraries:** torch (CPU build), transformers, peft, trl, accelerate, datasets, sentencepiece  
**No GPU required. No compilation step.** The pre-trained weights are loaded from disk, updated in-place by the optimizer during training, and written to disk as new weight files. The original checkpoint is never modified.

**One-time Python environment setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows WSL2: source .venv/bin/activate
                                   # Windows native: .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft trl accelerate sentencepiece
```

**Pinned minimum versions (`requirements.txt`):**

```
torch>=2.3.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0
trl>=0.8.6
accelerate>=0.29.0
sentencepiece>=0.1.99
```

After the first successful install, generate a full lockfile for reproducibility:

```bash
pip freeze > requirements.lock
```

Commit `requirements.lock` to version control. New team members install with `pip install -r requirements.lock` to get identical versions.

Total installed size: approximately 400–500 MB.

**WSL2 path note:** All paths passed to `slmkg train` and `slmkg eval` must be Linux-style paths (forward slashes, no drive letters). The C++ `std::system()` call constructs paths relative to the project root. If you open the project from Windows Explorer, use the WSL2 mount path (e.g., `/mnt/c/Users/.../project`) rather than the Windows UNC path. Mixing path styles will cause the Python scripts invoked by the C++ CLI to fail with a file-not-found error.

**Download GPT-2 checkpoints (one-time, no account required):**

```bash
python scripts/download_checkpoint.py \
  --checkpoint openai-community/gpt2 \
  --output checkpoints/base/gpt2-small

python scripts/download_checkpoint.py \
  --checkpoint openai-community/gpt2-medium \
  --output checkpoints/base/gpt2-medium
```

Downloads approximately 550 MB (small) and 1.4 GB (medium). Stored locally and never re-downloaded.

**Verify checkpoint loads and generates text:**

```bash
python3 -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('checkpoints/base/gpt2-small')
tok = GPT2Tokenizer.from_pretrained('checkpoints/base/gpt2-small')
ids = tok.encode('Knowledge graphs represent', return_tensors='pt')
out = model.generate(ids, max_new_tokens=20)
print(tok.decode(out[0]))
print('Parameters:', sum(p.numel() for p in model.parameters()))
"
```

**B1 fine-tuning run (natural text baseline, no KG data):**

```bash
./build/slmkg train sft \
  --base-checkpoint checkpoints/base/gpt2-small \
  --dataset data/datasets/pretrain/pretrain_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/pretrain.yaml \
  --output checkpoints/b1_baseline_small
```

Expected wall-clock time on a modern CPU: 20–40 minutes for a 5,000-example dataset with GPT-2 Small, LoRA rank 8, 3 epochs. GPT-2 Medium takes approximately 3× longer.

---

## 23.9 Phases 6 and 7: KG-Augmented SLM and RL

**Language:** Python 3.10+  
**Core libraries:** same virtual environment as Phase 5. No additional installs required.  
**No GPU required. No compilation step.**

**Phase 6 — KG-grounded SFT (baseline B4):**

```bash
source .venv/bin/activate
./build/slmkg train sft \
  --base-checkpoint checkpoints/base/gpt2-small \
  --dataset data/datasets/sft/sft_v0.1.jsonl \
  --model-config configs/model.yaml \
  --training-config configs/sft.yaml \
  --output checkpoints/b4_kg_sft_small
```

**Phase 7 — Preference optimization / DPO (baseline B5):**

```bash
./build/slmkg train dpo \
  --base-checkpoint checkpoints/b4_kg_sft_small \
  --preference-dataset data/datasets/preference/pref_v0.1.jsonl \
  --training-config configs/dpo.yaml \
  --output checkpoints/b5_kg_dpo_small
```

**Compare all baselines:**

```bash
./build/slmkg benchmark compare \
  --runs reports/eval_b0.json \
         reports/eval_b1_baseline.json \
         reports/eval_b2_linearized.json \
         reports/eval_b3_verbalized.json \
         reports/eval_b4_kg_sft.json \
         reports/eval_b5_kg_dpo.json \
  --output reports/benchmark_comparison.md
```

Run all development experiments with GPT-2 Small first. Repeat final experiments with GPT-2 Medium by substituting `checkpoints/base/gpt2-medium` as the base checkpoint. All scripts, configs, and dataset artifacts are identical between the two checkpoints; only the `--base-checkpoint` argument changes.

---

# 24. Implementation Plan

This section provides a task-level implementation plan for each phase. Tasks are numbered within each phase and listed in recommended build order. Estimated effort is in working days for a solo developer. Dependencies on prior tasks are noted where a task cannot begin before another completes.

Cross-cutting decisions that apply across all C++ phases:

- The `slmkg` binary is the single CLI entry point, built from `src/cli/main.cpp`.
- All C++ phases share the same CMake project. Adding a new phase adds new source files and new test targets; it does not restructure the build.
- The `slmkg train`, `slmkg eval`, and `slmkg benchmark` commands invoke Python scripts in `scripts/` via `std::system()`. The C++ CLI constructs the full command string, checks that `.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows) exists, calls `std::system()`, and treats a non-zero exit code as a hard failure. The training manifest is written by the Python script, not by the C++ CLI.
- All JSONL writers must flush after each record so that partial outputs are readable if a run is interrupted.

---

## 24.1 Phase 0: Specification and Fixtures

Estimated effort: 1–2 days.

| Task | Description | Depends on |
|---|---|---|
| T0.1 | Create full project directory structure (`mkdir -p ...`) | — |
| T0.2 | Write fixture corpus: 3–5 short `.md` files covering a narrow domain | T0.1 |
| T0.3 | Write `configs/kg_schema.yaml`: 3 entity types, 3 relation types, required fields | T0.1 |
| T0.4 | Write `tests/fixtures/mock_extraction_response.json`: one entity, one relation, matching Section 8 schema | T0.3 |
| T0.5 | Write `tests/fixtures/golden/rendered_prompt.txt`: expected output of prompt renderer for fixture chunk | T0.2, T0.3 |
| T0.6 | Write `tests/fixtures/golden/linearized_triple.txt`: expected triple linearization output | T0.3, T0.4 |
| T0.7 | Validate all YAML and JSON fixtures parse without error | T0.3, T0.4 |

**First verifiable milestone:** All fixture files exist, parse cleanly, and the directory structure matches Section 5.3.

---

## 24.2 Phase 1: Corpus and Prompt Infrastructure

Estimated effort: 8–10 days.

| Task | Description | Depends on |
|---|---|---|
| T1.1 | CMake project setup: `CMakeLists.txt` with FetchContent for nlohmann/json, yaml-cpp, Catch2 | T0.1 |
| T1.2 | `core/errors.hpp`: error types and result wrapper | T1.1 |
| T1.3 | `core/hashing.hpp`: SHA256 wrapper over OpenSSL | T1.1 |
| T1.4 | `core/config.hpp`: YAML config loader returning typed structs | T1.1, T1.2 |
| T1.5 | `corpus/document.hpp`: DocumentRecord struct and JSONL serialisation | T1.2 |
| T1.6 | `corpus/corpus_reader.hpp`: reads `.txt`, `.md`, `.jsonl` files into DocumentRecords | T1.5 |
| T1.7 | `corpus/chunker.hpp`: fixed-size chunker; `token_count` estimated as `char_count / 4` | T1.5 |
| T1.8 | `corpus/corpus_validator.hpp`: checks required fields, duplicate IDs, encoding | T1.5, T1.6 |
| T1.9 | `corpus/corpus_manifest.hpp`: generates manifest YAML; hashes the full chunks.jsonl content | T1.6, T1.7, T1.3 |
| T1.10 | `prompts/prompt_template.hpp`: loads prompt file and extracts `{{variable}}` placeholders | T1.2 |
| T1.11 | `prompts/prompt_renderer.hpp`: substitutes template variables; errors on missing variables | T1.10 |
| T1.12 | `prompts/prompt_hash.hpp`: SHA256 of rendered prompt string | T1.11, T1.3 |
| T1.13 | `prompts/prompt_manifest.hpp`: records paths, hashes, and template variable values | T1.12 |
| T1.14 | CLI: `slmkg corpus ingest`, `slmkg corpus validate`, `slmkg corpus stats` | T1.8, T1.9 |
| T1.15 | CLI: `slmkg prompt render`, `slmkg prompt hash`, `slmkg prompt validate` | T1.11, T1.12, T1.13 |
| T1.16 | Phase 1 unit tests (see Section 25.3) | T1.14, T1.15 |

**First verifiable milestone:** `slmkg corpus ingest` produces `corpus_manifest.yaml` and `chunks.jsonl` from the fixture corpus. All Phase 1 unit tests pass.

---

## 24.3 Phase 2: KG Extraction Layer

Estimated effort: 10–12 days.

| Task | Description | Depends on |
|---|---|---|
| T2.1 | `extraction/provider_interface.hpp`: `IKgExtractionProvider` pure virtual interface | T1.2 |
| T2.2 | `extraction/response_parser.hpp`: validates JSON response against extraction output schema | T2.1 |
| T2.3 | `extraction/mock_provider.hpp`: returns fixture response for known chunk IDs; errors for unknown | T2.1 |
| T2.4 | `extraction/extraction_cache.hpp`: file-based cache keyed on rendered-prompt SHA256 | T1.3, T2.1 |
| T2.5 | `extraction/replay_provider.hpp`: loads from cache directory; fails if cache miss | T2.4 |
| T2.6 | `extraction/openai_provider.hpp`: HTTP POST to OpenAI chat completions endpoint via libcurl | T2.1 |
| T2.7 | `extraction/gemini_provider.hpp`: HTTP POST to Gemini generateContent endpoint via libcurl | T2.1 |
| T2.8 | Retry logic in providers: exponential backoff, `max_attempts` from config | T2.6, T2.7 |
| T2.9 | `extraction/extraction_pipeline.hpp`: iterates chunks, calls provider, writes call records and candidate JSONL | T2.2, T2.4, T2.8 |
| T2.10 | CLI: `slmkg kg extract` with `--mock` and `--replay` flags; writes `.cache/kg_extraction/` | T2.9 |
| T2.11 | Phase 2 unit and integration tests; all CI tests use mock or replay provider | T2.10 |

**First verifiable milestone:** `slmkg kg extract --mock` processes the fixture corpus and writes candidate entities and relations to `data/kg/kg_v0.1/` without making any API calls.

---

## 24.4 Phase 3: KG Normalization and Validation

Estimated effort: 6–8 days.

| Task | Description | Depends on |
|---|---|---|
| T3.1 | `kg/schema.hpp`: loads `kg_schema.yaml`; validates entity and relation type lists | T1.4 |
| T3.2 | `kg/entity.hpp`: CanonicalEntity struct and JSONL serialisation | T1.2 |
| T3.3 | `kg/relation.hpp`: CanonicalRelation struct and JSONL serialisation | T1.2 |
| T3.4 | `kg/graph_store.hpp`: in-memory store with JSONL read/write for entities and relations | T3.2, T3.3 |
| T3.5 | `kg/normalizer.hpp`: entity normalizer using lowercase + whitespace-strip + exact-string match for Phase 1; merges candidates with identical canonical names | T3.2, T3.4 |
| T3.6 | `kg/normalizer.hpp`: alias resolver; records all known aliases per canonical entity | T3.5 |
| T3.7 | `kg/graph_validator.hpp`: schema constraint checks; domain/range checks; writes rejected candidates | T3.1, T3.3 |
| T3.8 | `kg/graph_snapshot.hpp`: sorted serialisation + SHA256 snapshot hash; writes `manifest.yaml` | T3.4, T1.3 |
| T3.9 | CLI: `slmkg kg normalize`, `slmkg kg validate`, `slmkg kg snapshot` | T3.5, T3.7, T3.8 |
| T3.10 | Phase 3 unit tests | T3.9 |

**First verifiable milestone:** `slmkg kg normalize` then `slmkg kg validate` on mock-extracted candidates produces a validation report with zero hard errors and a snapshot manifest with a stable hash.

**Note on normalization:** Phase 3 uses string-based normalization only (lowercase, whitespace-strip, exact match). Embedding-based or fuzzy normalization is deferred to a later extension. This is a known simplification and should be flagged in evaluation when alias quality is measured.

---

## 24.5 Phase 4: Dataset Generation

Estimated effort: 6–8 days.

| Task | Description | Depends on |
|---|---|---|
| T4.1 | `kg/graph_linearizer.hpp`: triple linearization, path linearization, natural-language verbalization | T3.4 |
| T4.2 | `datasets/negative_sampler.hpp`: corrupt relation type or entity type using schema constraints; verify generated triple is not a true fact | T3.4, T3.1 |
| T4.3 | `datasets/pretrain_builder.hpp`: mixes natural text chunks with linearized and verbalized triples per configured ratios | T4.1, T1.7 |
| T4.4 | `datasets/sft_builder.hpp`: single-hop QA and multi-hop QA from graph facts and paths | T4.1 |
| T4.5 | `datasets/sft_builder.hpp`: missing-evidence refusal and contradiction detection task types | T4.4 |
| T4.6 | Length filter in SFT builder: serialize each record to the GPT-2 prompt template format and reject records whose total character estimate exceeds 3600 characters (~900 tokens, leaving 124 tokens of headroom within the 1024-token context limit) | T4.4, T4.5 |
| T4.7 | `datasets/preference_builder.hpp`: chosen (graph-consistent) vs rejected (graph-inconsistent) pairs | T4.2 |
| T4.8 | `datasets/eval_builder.hpp`: holds out a fixed KG slice before generating any training data | T3.4 |
| T4.9 | Dataset manifest generator: record counts, token estimates, KG snapshot ID, content hash | T4.3, T4.4, T4.7, T4.8 |
| T4.10 | CLI: all `slmkg dataset` commands | T4.9 |
| T4.11 | Phase 4 unit and golden-file tests | T4.10 |

**First verifiable milestone:** `slmkg dataset build-sft` produces a non-empty `sft_v0.1.jsonl` from the fixture KG snapshot, each record within the 1024-token budget, with no true fact appearing as a negative sample.

---

## 24.6 Phase 5: Baseline SLM

Estimated effort: 4–5 days.

| Task | Description | Depends on |
|---|---|---|
| T5.1 | `requirements.txt` and `scripts/setup_env.sh`: creates `.venv`, installs CPU torch and HuggingFace stack | — |
| T5.2 | `scripts/download_checkpoint.py`: downloads checkpoint to a local path; verifies file integrity; prints parameter count | T5.1 |
| T5.3 | `scripts/extend_tokenizer.py`: adds KG special tokens, resizes embeddings, saves extended tokenizer to `checkpoints/tokenizer_kg_v0.1/` | T5.2 |
| T5.4 | `scripts/finetune_sft.py`: LoRA SFT via `trl.SFTTrainer`; reads config from YAML; writes training manifest on completion | T5.3 |
| T5.5 | `scripts/evaluate_checkpoint.py`: loads checkpoint, runs inference on eval JSONL, writes metrics to JSON; called by `slmkg eval run` via `std::system()` | T5.4 |
| T5.6 | C++ CLI: `slmkg train download`, `slmkg train sft` — construct Python command strings and invoke via `std::system()`; validate `.venv` exists before calling | T5.4 |
| T5.7 | C++ CLI: `slmkg eval run` — invoke `scripts/evaluate_checkpoint.py` via `std::system()` | T5.5 |
| T5.8 | Run B0 (zero-shot, frozen checkpoint) and B1 (natural-text SFT) and confirm reports differ | T5.7 |

**First verifiable milestone:** `slmkg train sft` completes without error, writes a checkpoint and a training manifest, and `slmkg eval run` produces a JSON report with numeric metrics.

---

## 24.7 Phase 6: KG-Augmented SLM

Estimated effort: 3–4 days.

| Task | Description | Depends on |
|---|---|---|
| T6.1 | Configure `configs/sft.yaml` to point to KG-derived SFT dataset from Phase 4 | T4.10, T5.4 |
| T6.2 | Run B2 (graph-linearized continued pre-training) | T6.1 |
| T6.3 | Run B3 (graph-verbalized continued pre-training) | T6.1 |
| T6.4 | Run B4 (KG-grounded SFT) | T6.1 |
| T6.5 | Run `slmkg benchmark compare` across B0–B4; document ablation findings | T6.4 |

**First verifiable milestone:** B4 benchmark report shows a measurable difference from B1 on at least one KG-specific metric (graph QA accuracy or contradiction rate).

---

## 24.8 Phase 7: KG-Based RL

Estimated effort: 4–5 days.

| Task | Description | Depends on |
|---|---|---|
| T7.1 | `scripts/finetune_dpo.py`: DPO via `trl.DPOTrainer`; reads preference JSONL from Phase 4 | T5.4 |
| T7.2 | `scripts/evaluate_kg_reward.py`: computes per-record reward components (fact, consistency, reasoning, uncertainty, language); writes reward breakdown to JSON | T5.5 |
| T7.3 | C++ CLI: `slmkg train dpo` — invoke `scripts/finetune_dpo.py` via `std::system()` | T7.1 |
| T7.4 | Run B5 (DPO) and evaluate with `slmkg eval run` | T7.3 |
| T7.5 | Run full `slmkg benchmark compare` across B0–B5; document reward overfitting analysis | T7.4 |

**First verifiable milestone:** B5 benchmark report is generated and the DPO run completes without reward collapse (total reward does not monotonically increase while graph QA accuracy falls).

---

## 24.9 Effort Summary

### 24.9.1 Solo Developer Baseline

These are the original estimates for a solo developer writing code without AI assistance.

| Phase | Description | Estimated days (solo) |
|---|---|---|
| Phase 0 | Fixtures | 1–2 |
| Phase 1 | Corpus and prompt infrastructure | 8–10 |
| Phase 2 | KG extraction layer | 10–12 |
| Phase 3 | KG normalization and validation | 6–8 |
| Phase 4 | Dataset generation | 6–8 |
| Phase 5 | Baseline SLM | 4–5 |
| Phase 6 | KG-augmented SLM | 3–4 |
| Phase 7 | KG-based RL | 4–5 |
| **Total** | | **42–54 days** |

### 24.9.2 AI-Assisted Estimate (Claude for coding, human for debugging and testing)

With Claude generating all code from the spec, the bottleneck shifts from writing code to reviewing it, debugging build and integration issues, running tests, and interpreting results. The speedup is not uniform across tasks.

**Where AI assistance gives the largest speedup (4–5×):**

- C++ struct definitions, JSONL serializers, YAML config loaders — Claude generates these directly from the spec's data schemas in minutes.
- Python HuggingFace scripts (`finetune_sft.py`, `finetune_dpo.py`, `evaluate_checkpoint.py`) — Claude knows the `trl`, `peft`, and `transformers` APIs well.
- Catch2 and pytest test files — Claude writes tests from function signatures and expected behaviour descriptions.
- Prompt renderer and hash logic — well-defined string operations with clear input/output contracts.

**Where AI assistance gives a moderate speedup (2–3×):**

- libcurl and OpenSSL integration — Claude generates correct code, but platform-specific linker and header issues on WSL2 require human diagnosis.
- CMake build system — Claude writes the `CMakeLists.txt`, but resolving FetchContent failures or missing system packages still requires human attention.
- CLI argument parsing — mechanical but requires human testing of all flag combinations.

**Where AI assistance gives a small speedup (1–1.5×) or none:**

- Training wall-clock time — GPT-2 fine-tuning on CPU takes the same time regardless of how the code was written. This is a hard constraint.
- KG quality assessment — after running on a real corpus, deciding whether the extracted graph is meaningful is human judgment.
- Benchmark interpretation — comparing B0–B5 and drawing research conclusions is the core PhD work.
- Debugging training instability — if loss curves diverge or produce NaN, the diagnostic loop (run → observe → hypothesize → fix → re-run) requires human judgment even when Claude suggests fixes.

**Revised effort table (AI-assisted coding, human debugging and testing):**

| Phase | Solo days | AI-assisted days | Notes |
|---|---|---|---|
| Phase 0 | 1–2 | 0.5 | Claude generates all fixtures from spec schemas |
| Phase 1 | 8–10 | 2–3 | CMake + structs + CLI; human resolves WSL2 build issues |
| Phase 2 | 10–12 | 3–4 | libcurl setup is the main human debugging cost |
| Phase 3 | 6–8 | 1–2 | Straightforward data transforms; human validates normalization quality |
| Phase 4 | 6–8 | 1–2 | Builders and linearizers map directly from spec; human validates golden files |
| Phase 5 | 4–5 | 1–2 | HuggingFace scripts fast to generate; human monitors training runs |
| Phase 6 | 3–4 | 0.5–1 | Config changes and training runs; human monitors and interprets |
| Phase 7 | 4–5 | 1–2 | DPO script fast; human checks for reward collapse |
| **Total coding** | **42–54** | **10–17** | |

**Non-compressible time (independent of coding speed):**

| Activity | Wall-clock time | Notes |
|---|---|---|
| B0–B5 training runs, GPT-2 Small | 4–8 hours per run × 6 runs | Runs overnight; no human attention needed while running |
| B0–B5 training runs, GPT-2 Medium | 12–18 hours per run × 6 runs | Final experiments only; scheduled as overnight jobs |
| KG extraction on real corpus (live API) | 1–4 hours | Depends on corpus size and API concurrency limit |
| KG quality review | 2–4 hours | Human reviews validation report and sample entities |
| Benchmark result interpretation | Ongoing | Research work; no time compression |

**Total human engagement: 12–21 working days** (coding review + debugging + test runs + monitoring + interpretation), plus training wall-clock time that runs unsupervised overnight.

At a part-time PhD pace of approximately two focused days per week, the full pipeline is a **6–11 week project** for Phase 0 through first benchmark results (B0–B5 with GPT-2 Small).

This compares to the solo baseline of 21–27 weeks for the same scope.

The GPT-2 Medium final experiments add approximately 3–4 additional weeks of calendar time at part-time pace, primarily because of overnight training run scheduling rather than additional coding.

---

# 25. Testing Plan

Section 15 lists all required test names organized by module. This section specifies the test infrastructure, run commands, CI strategy, golden file workflow, and coverage expectations.

---

## 25.1 Test Infrastructure

**C++ tests:** Catch2 v3, run via ctest. Test source files live in `tests/`. Each module has a corresponding test file, for example `tests/test_corpus_reader.cpp`.

**Python tests:** pytest, run from the project root. Python test files live in `tests/python/`. Each script in `scripts/` has a corresponding test file, for example `tests/python/test_finetune_sft.py`.

**Fixtures:** All test fixtures live in `tests/fixtures/`. Golden files live in `tests/fixtures/golden/`. Recorded API replay files live in `tests/fixtures/replay/`.

**No live API calls in any automated test.** All extraction tests use the mock provider or the replay provider. API keys must never be present in the test environment.

---

## 25.2 C++ Test Runner

**Run all C++ tests:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

**Run a specific test group:**

```bash
ctest --test-dir build -R "CorpusReader" --output-on-failure
ctest --test-dir build -R "PromptRenderer|PromptHash" --output-on-failure
ctest --test-dir build -R "MockProvider|ReplayProvider" --output-on-failure
ctest --test-dir build -R "KgSchema|GraphStore" --output-on-failure
ctest --test-dir build -R "SftBuilder|GraphLinearizer" --output-on-failure
```

**Run only fast unit tests (excluding integration tests):**

```bash
ctest --test-dir build -L unit --output-on-failure
```

**Run only integration tests:**

```bash
ctest --test-dir build -L integration --output-on-failure
```

Tag each Catch2 test case with `[unit]` or `[integration]` so ctest label filtering works.

---

## 25.3 Python Test Runner

**Setup:**

```bash
source .venv/bin/activate
pip install pytest pytest-cov
```

**Run all Python tests:**

```bash
pytest tests/python/ -v
```

**Run with coverage:**

```bash
pytest tests/python/ --cov=scripts --cov-report=term-missing
```

**Key Python test files:**

| Test file | Covers |
|---|---|
| `tests/python/test_download_checkpoint.py` | Checkpoint download and file integrity verification |
| `tests/python/test_extend_tokenizer.py` | KG special token addition, embedding resize, mean initialization |
| `tests/python/test_finetune_sft.py` | SFT training loop runs without error on a 10-record fixture dataset |
| `tests/python/test_evaluate_checkpoint.py` | Evaluation script produces valid JSON output from a fixture checkpoint |
| `tests/python/test_finetune_dpo.py` | DPO training loop runs without error on a 10-record fixture preference dataset |

Python tests must use a local fixture checkpoint (a randomly initialized GPT-2 Small with the same architecture) rather than downloading the real pre-trained checkpoint. This keeps tests fast and offline.

Create a fixture checkpoint once:

```bash
python3 -c "
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
config = GPT2Config(n_layer=2, n_head=2, n_embd=64)
model = GPT2LMHeadModel(config)
model.save_pretrained('tests/fixtures/gpt2_tiny')
GPT2Tokenizer.from_pretrained('openai-community/gpt2').save_pretrained('tests/fixtures/gpt2_tiny')
"
```

All Python test scripts should point to `tests/fixtures/gpt2_tiny` rather than `checkpoints/base/gpt2-small`.

---

## 25.4 Test Categories

| Category | Description | Runs in CI | API keys needed |
|---|---|---|---|
| Unit | Single-module, no file I/O beyond fixtures | Yes | No |
| Integration | Multi-module pipeline, fixture corpus and KG | Yes | No |
| Golden-file | Compare output to committed golden files | Yes | No |
| Mock provider | Full extraction pipeline using mock provider | Yes | No |
| Replay provider | Full extraction pipeline using recorded responses | Yes | No |
| Live provider | Real API calls to OpenAI or Gemini | No, manual only | Yes |
| Training smoke | One-epoch training run on tiny fixture dataset | Yes (Python) | No |
| Benchmark smoke | Benchmark runner produces valid JSON report | Yes (Python) | No |

---

## 25.5 CI Strategy

The CI pipeline must complete with no live API calls and no pre-trained checkpoint downloads.

**Required CI steps in order:**

```
1. cmake -B build -DCMAKE_BUILD_TYPE=Debug
2. cmake --build build --parallel
3. ctest --test-dir build -L unit --output-on-failure
4. ctest --test-dir build -L integration --output-on-failure
5. ctest --test-dir build -L golden --output-on-failure
6. source .venv/bin/activate
7. pytest tests/python/ -v
```

Steps 1–5 cover all C++ components. Steps 6–7 cover all Python scripts using the tiny fixture checkpoint.

**CI must fail if:**

- any unit test fails,
- any integration test fails,
- any golden file comparison differs from the committed reference,
- any Python test fails,
- any test requires an API key (`OPENAI_API_KEY` or `GEMINI_API_KEY` must not be set in CI),
- any test downloads a file from the internet.

---

## 25.6 Golden File Workflow

Golden files are the committed reference outputs for deterministic pipeline steps: rendered prompts, graph linearizations, dataset record formats, manifest structures, and validation reports.

**When to update golden files:**

- A deliberate change to a prompt renderer, linearizer, or manifest format requires a golden file update.
- Accidental golden file changes indicate a regression and must not be committed.

**Update procedure:**

```bash
# Re-run the specific golden test with overwrite enabled
ctest --test-dir build -R "GoldenPromptRender" -D GOLDEN_OVERWRITE=1

# Review the diff before committing
git diff tests/fixtures/golden/
```

Only commit golden file changes when the diff is fully understood and intentional. Include the reason in the commit message.

---

## 25.7 API Isolation

The extraction pipeline must never make live API calls during automated tests. This is enforced at two levels.

**Code level:** Every test that instantiates an extraction pipeline must pass a `MockKgExtractionProvider` or `ReplayKgExtractionProvider` explicitly. The `OpenAIKgExtractionProvider` and `GeminiKgExtractionProvider` constructors should assert that `SLMKG_TEST_MODE` is not set, or alternatively the test harness can unset `OPENAI_API_KEY` and `GEMINI_API_KEY` before running.

**Configuration level:** Add a `test_mode: true` flag to `kg_extraction.yaml` that forces the mock provider regardless of the `provider:` setting. CI always runs with this flag set.

**Recording new replay fixtures:**

When a new extraction prompt is added and replay fixtures are needed, run once with the live provider and save the responses:

```bash
export OPENAI_API_KEY=sk-...
./build/slmkg kg extract \
  --corpus tests/fixtures/fixture_corpus \
  --schema tests/fixtures/kg_schema.yaml \
  --system-prompt prompts/kg_system_prompt_v0.1.txt \
  --user-prompt prompts/kg_user_prompt_v0.1.txt \
  --output /tmp/kg_extract_tmp \
  --config configs/kg_extraction.yaml \
  --record-to tests/fixtures/replay/
```

Commit the recorded responses to `tests/fixtures/replay/`. Subsequent CI runs use the replay provider against these files.

---

## 25.8 Coverage Expectations

**C++ coverage targets:**

| Module | Minimum coverage expectation |
|---|---|
| Corpus reader | All supported file types (.txt, .md, .jsonl) exercised |
| Chunker | Both min and max chunk size boundaries tested |
| Prompt renderer | Every template variable tested; missing variable tested |
| Response parser | Valid response, invalid JSON, schema violation, empty array all tested |
| Entity normalizer | Duplicate canonical name merge, alias resolution, rejection tested |
| Graph validator | Domain violation, range violation, unknown type, valid triple all tested |
| SFT builder | Every task type (single-hop QA, multi-hop QA, refusal, contradiction) tested |
| Negative sampler | Verified no generated negative matches a true fact from the KG |

**Python coverage targets:**

| Script | Minimum coverage expectation |
|---|---|
| `finetune_sft.py` | Training loop runs to completion on tiny fixture dataset |
| `extend_tokenizer.py` | All 7 special tokens present in extended vocabulary |
| `evaluate_checkpoint.py` | All required metric fields present in output JSON |
| `finetune_dpo.py` | DPO loop runs to completion on tiny fixture preference dataset |
