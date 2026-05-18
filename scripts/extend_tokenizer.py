#!/usr/bin/env python3
"""Extend a GPT-2 tokenizer with KG special tokens and resize model embeddings."""

import argparse
import json
import os
import sys


KG_SPECIAL_TOKENS = [
    "[ENTITY]",
    "[RELATION]",
    "[GRAPH_START]",
    "[GRAPH_END]",
    "[HOP]",
    "[EVIDENCE]",
]


def extend_tokenizer(base_checkpoint: str, output: str, model_checkpoint: str = "") -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output, exist_ok=True)

    print(f"[extend] Loading tokenizer from: {base_checkpoint}")
    tok = AutoTokenizer.from_pretrained(base_checkpoint)

    original_vocab_size = len(tok)
    print(f"[extend] Original vocabulary size: {original_vocab_size}")

    new_tokens = [t for t in KG_SPECIAL_TOKENS if t not in tok.get_vocab()]
    if not new_tokens:
        print("[extend] All KG tokens already present, nothing to add.")
    else:
        tok.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"[extend] Added {len(new_tokens)} tokens: {new_tokens}")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"[extend] Set pad_token = eos_token ({tok.eos_token!r})")

    tok.save_pretrained(output)
    print(f"[extend] Tokenizer saved to: {output}")
    print(f"[extend] New vocabulary size: {len(tok)}")

    if model_checkpoint:
        print(f"[extend] Loading model from: {model_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        model.resize_token_embeddings(len(tok))
        model.save_pretrained(output)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[extend] Model saved to: {output} ({param_count:,} parameters)")

    manifest = {
        "base_checkpoint": base_checkpoint,
        "output": output,
        "original_vocab_size": original_vocab_size,
        "new_vocab_size": len(tok),
        "added_tokens": new_tokens,
    }
    manifest_path = os.path.join(output, "extension_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extend] Manifest written to: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add KG special tokens to a tokenizer and optionally resize model embeddings."
    )
    parser.add_argument("--base-checkpoint", required=True,
                        help="Base tokenizer directory (e.g. checkpoints/base/gpt2-small)")
    parser.add_argument("--output", required=True,
                        help="Output directory for the extended tokenizer")
    parser.add_argument("--model-checkpoint", default="",
                        help="If provided, also resize model embeddings and save here")
    args = parser.parse_args()

    extend_tokenizer(args.base_checkpoint, args.output, args.model_checkpoint)


if __name__ == "__main__":
    main()
