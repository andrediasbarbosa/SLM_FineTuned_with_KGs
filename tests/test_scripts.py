"""
Unit tests for Phase 5 Python scripts.

Tests only pure logic (no model downloads, no training runs):
  - scripts/finetune_sft.py  — format_record, load_jsonl
  - scripts/evaluate_checkpoint.py — token_f1, exact_match, build_prompt, load_jsonl
  - scripts/extend_tokenizer.py   — KG_SPECIAL_TOKENS list
  - scripts/download_checkpoint.py — _verify_checkpoint error paths

Run with:
  .venv/bin/pytest tests/test_scripts.py -v
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import helpers — load scripts as modules without executing their main()
# ---------------------------------------------------------------------------
SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


finetune   = _load("finetune_sft")
evaluate   = _load("evaluate_checkpoint")
extend_tok = _load("extend_tokenizer")
download   = _load("download_checkpoint")


# ---------------------------------------------------------------------------
# finetune_sft — format_record
# ---------------------------------------------------------------------------
class TestFormatRecord:
    def test_uses_text_field_when_present(self):
        rec = {"text": "pre-formatted text", "input": "ignored", "output": "ignored"}
        assert finetune.format_record(rec) == "pre-formatted text"

    def test_builds_from_instruction_input_output(self):
        rec = {
            "instruction": "Answer using the graph.",
            "input": "What does A support?",
            "output": "A supports B.",
        }
        result = finetune.format_record(rec)
        assert "Answer using the graph." in result
        assert "Question: What does A support?" in result
        assert "Answer: A supports B." in result

    def test_includes_graph_context_when_present(self):
        rec = {
            "instruction": "Use evidence.",
            "graph_context": ["[ENTITY] A [RELATION] supports [ENTITY] B"],
            "input": "Question?",
            "output": "Answer.",
        }
        result = finetune.format_record(rec)
        assert "[ENTITY] A [RELATION] supports [ENTITY] B" in result

    def test_skips_empty_graph_context(self):
        rec = {
            "instruction": "Use evidence.",
            "graph_context": [],
            "input": "Question?",
            "output": "Answer.",
        }
        result = finetune.format_record(rec)
        assert "Graph context" not in result

    def test_missing_instruction_still_works(self):
        rec = {"input": "Q?", "output": "A."}
        result = finetune.format_record(rec)
        assert "Question: Q?" in result
        assert "Answer: A." in result

    def test_empty_record_returns_empty_string(self):
        result = finetune.format_record({})
        assert result == ""


# ---------------------------------------------------------------------------
# finetune_sft — load_jsonl
# ---------------------------------------------------------------------------
class TestFinetuneLoadJsonl:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        records = [{"id": "r1", "text": "hello"}, {"id": "r2", "text": "world"}]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = finetune.load_jsonl(str(p))
        assert len(loaded) == 2
        assert loaded[0]["id"] == "r1"
        assert loaded[1]["text"] == "world"

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"id":"r1"}\n\n{"id":"r2"}\n')
        loaded = finetune.load_jsonl(str(p))
        assert len(loaded) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert finetune.load_jsonl(str(p)) == []


# ---------------------------------------------------------------------------
# evaluate_checkpoint — token_f1
# ---------------------------------------------------------------------------
class TestTokenF1:
    def test_identical_strings_score_one(self):
        assert evaluate.token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_strings_score_zero(self):
        assert evaluate.token_f1("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = evaluate.token_f1("a b c", "a b d")
        assert 0.0 < score < 1.0

    def test_empty_pred_scores_zero(self):
        assert evaluate.token_f1("", "reference") == pytest.approx(0.0)

    def test_empty_gold_scores_zero(self):
        assert evaluate.token_f1("prediction", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert evaluate.token_f1("The Cat", "the cat") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_checkpoint — exact_match
# ---------------------------------------------------------------------------
class TestExactMatch:
    def test_identical_strings_match(self):
        assert evaluate.exact_match("hello world", "hello world") is True

    def test_case_insensitive(self):
        assert evaluate.exact_match("Hello", "hello") is True

    def test_leading_trailing_whitespace_ignored(self):
        assert evaluate.exact_match("  hello  ", "hello") is True

    def test_different_strings_do_not_match(self):
        assert evaluate.exact_match("cat", "dog") is False


# ---------------------------------------------------------------------------
# evaluate_checkpoint — build_prompt
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_contains_instruction(self):
        rec = {"instruction": "Answer with evidence.", "input": "Q?"}
        prompt = evaluate.build_prompt(rec)
        assert "Answer with evidence." in prompt

    def test_contains_input(self):
        rec = {"instruction": "Instr.", "input": "What is X?"}
        prompt = evaluate.build_prompt(rec)
        assert "What is X?" in prompt

    def test_ends_with_answer_marker(self):
        rec = {"instruction": "Instr.", "input": "Q?"}
        prompt = evaluate.build_prompt(rec)
        assert prompt.strip().endswith("Answer:")

    def test_includes_graph_context_when_present(self):
        rec = {
            "instruction": "Instr.",
            "graph_context": ["[ENTITY] A [RELATION] r [ENTITY] B"],
            "input": "Q?",
        }
        prompt = evaluate.build_prompt(rec)
        assert "[ENTITY] A" in prompt

    def test_no_graph_context_key_still_works(self):
        rec = {"instruction": "Instr.", "input": "Q?"}
        prompt = evaluate.build_prompt(rec)
        assert "Graph context" not in prompt


# ---------------------------------------------------------------------------
# evaluate_checkpoint — load_jsonl (shared helper, same contract)
# ---------------------------------------------------------------------------
class TestEvaluateLoadJsonl:
    def test_loads_eval_record(self, tmp_path):
        p = tmp_path / "eval.jsonl"
        rec = {"id": "eval_000001", "input": "Q?", "expected_output": "A."}
        p.write_text(json.dumps(rec) + "\n")
        loaded = evaluate.load_jsonl(str(p))
        assert loaded[0]["expected_output"] == "A."


# ---------------------------------------------------------------------------
# extend_tokenizer — KG_SPECIAL_TOKENS
# ---------------------------------------------------------------------------
class TestKgSpecialTokens:
    def test_required_tokens_present(self):
        required = {"[ENTITY]", "[RELATION]", "[GRAPH_START]", "[GRAPH_END]"}
        assert required.issubset(set(extend_tok.KG_SPECIAL_TOKENS))

    def test_no_duplicates(self):
        tokens = extend_tok.KG_SPECIAL_TOKENS
        assert len(tokens) == len(set(tokens))

    def test_all_tokens_have_bracket_format(self):
        for t in extend_tok.KG_SPECIAL_TOKENS:
            assert t.startswith("[") and t.endswith("]"), f"Bad token format: {t}"


# ---------------------------------------------------------------------------
# download_checkpoint — _verify_checkpoint (error paths, no network)
# ---------------------------------------------------------------------------
class TestVerifyCheckpoint:
    def test_missing_config_json_raises(self, tmp_path):
        # only tokenizer_config.json, no config.json
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        with pytest.raises(SystemExit):
            download._verify_checkpoint(str(tmp_path))

    def test_empty_config_json_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        with pytest.raises(SystemExit):
            download._verify_checkpoint(str(tmp_path))

    def test_no_weight_files_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        with pytest.raises(SystemExit):
            download._verify_checkpoint(str(tmp_path))

    def test_valid_checkpoint_passes(self, tmp_path):
        (tmp_path / "config.json").write_text('{"model_type":"gpt2"}')
        (tmp_path / "tokenizer_config.json").write_text('{"model_type":"gpt2"}')
        (tmp_path / "model.safetensors").write_bytes(b"fake weights")
        download._verify_checkpoint(str(tmp_path))  # must not raise
