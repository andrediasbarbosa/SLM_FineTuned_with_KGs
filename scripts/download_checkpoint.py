#!/usr/bin/env python3
"""Download and verify a HuggingFace checkpoint to a local directory."""

import argparse
import hashlib
import os
import sys


def download_checkpoint(checkpoint: str, output: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output, exist_ok=True)

    print(f"[download] Downloading tokenizer: {checkpoint}")
    tok = AutoTokenizer.from_pretrained(checkpoint)
    tok.save_pretrained(output)

    print(f"[download] Downloading model: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model.save_pretrained(output)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[download] Saved to: {output}")
    print(f"[download] Parameters: {param_count:,}")

    _verify_checkpoint(output)


def _verify_checkpoint(path: str) -> None:
    """Check that the expected files exist and are non-empty."""
    required = ["config.json", "tokenizer_config.json"]
    for fname in required:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"[verify] MISSING: {fpath}", file=sys.stderr)
            sys.exit(1)
        if os.path.getsize(fpath) == 0:
            print(f"[verify] EMPTY: {fpath}", file=sys.stderr)
            sys.exit(1)

    weight_files = [
        f for f in os.listdir(path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    if not weight_files:
        print(f"[verify] No weight files found in {path}", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(os.path.getsize(os.path.join(path, f)) for f in weight_files)
    print(f"[verify] Weight files: {weight_files}")
    print(f"[verify] Total weight size: {total_bytes / 1e6:.1f} MB")
    print("[verify] OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a HuggingFace checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Local output directory")
    args = parser.parse_args()

    download_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
