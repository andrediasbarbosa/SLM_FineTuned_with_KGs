#!/usr/bin/env python3
"""Interactive prompt loop for one or two local checkpoints.

Single model:
  python scripts/prompt.py --checkpoint checkpoints/base/gpt2-small

Side-by-side comparison:
  python scripts/prompt.py \
    --checkpoint checkpoints/base/gpt2-small \
    --compare    checkpoints/b1_baseline_small

Flags:
  --checkpoint     <path>   Primary checkpoint (required)
  --compare        <path>   Second checkpoint for side-by-side output (optional)
  --temperature    <float>  Sampling temperature (default: 0.8)
  --top-p          <float>  Nucleus sampling p (default: 0.95)
  --max-new-tokens <int>    Max tokens to generate per prompt (default: 80)
  --no-sample               Use greedy decoding instead of sampling
"""

import argparse
import sys


def load_model(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print(f"[prompt] Loading {path} ...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"{params:,} parameters")
    return tok, model


def generate(tok, model, prompt: str, max_new_tokens: int,
             temperature: float, top_p: float, do_sample: bool) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(output_ids[0], skip_special_tokens=True)


def print_divider(char: str = "─", width: int = 72) -> None:
    print(char * width)


def run(args: argparse.Namespace) -> None:
    tok_a, model_a = load_model(args.checkpoint)
    label_a = args.checkpoint.rstrip("/").split("/")[-1]

    tok_b = model_b = label_b = None
    if args.compare:
        tok_b, model_b = load_model(args.compare)
        label_b = args.compare.rstrip("/").split("/")[-1]

    do_sample = not args.no_sample

    print()
    if model_b is not None:
        print(f"Comparing  [{label_a}]  vs  [{label_b}]")
    else:
        print(f"Model: [{label_a}]")
    print(f"Settings: max_new_tokens={args.max_new_tokens}  "
          f"temperature={args.temperature}  top_p={args.top_p}  "
          f"sampling={'on' if do_sample else 'off (greedy)'}")
    print("Type a prompt and press Enter. Ctrl+C or Ctrl+D to quit.\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[prompt] Bye.")
            break

        if not prompt:
            continue

        if model_b is None:
            # Single model
            output = generate(tok_a, model_a, prompt,
                              args.max_new_tokens, args.temperature,
                              args.top_p, do_sample)
            print_divider()
            print(output)
            print_divider()
        else:
            # Side-by-side
            out_a = generate(tok_a, model_a, prompt,
                             args.max_new_tokens, args.temperature,
                             args.top_p, do_sample)
            out_b = generate(tok_b, model_b, prompt,
                             args.max_new_tokens, args.temperature,
                             args.top_p, do_sample)
            print_divider("─")
            print(f"[{label_a}]")
            print(out_a)
            print_divider("·")
            print(f"[{label_b}]")
            print(out_b)
            print_divider("─")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive prompt loop for local HuggingFace checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Primary checkpoint directory")
    parser.add_argument("--compare", default="",
                        help="Second checkpoint for side-by-side comparison")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--no-sample", action="store_true",
                        help="Use greedy decoding instead of sampling")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
