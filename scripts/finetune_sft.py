#!/usr/bin/env python3
"""LoRA SFT fine-tuning via trl.SFTTrainer.

Reads training config and model config from YAML files.
Writes a training manifest JSON on completion.

Usage:
  python scripts/finetune_sft.py \
    --base-checkpoint checkpoints/base/gpt2-small \
    --dataset data/datasets/sft/sft_v0.1.jsonl \
    --model-config configs/model.yaml \
    --training-config configs/sft.yaml \
    --output checkpoints/b1_baseline_small
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_record(record: dict) -> str:
    """Format an SFT or pretrain record as a single training text string."""
    if "text" in record:
        return record["text"]

    parts = []
    if "instruction" in record:
        parts.append(record["instruction"])
    if "graph_context" in record and record["graph_context"]:
        parts.append("Graph context: " + " | ".join(record["graph_context"]))
    if "input" in record:
        parts.append("Question: " + record["input"])
    if "output" in record:
        parts.append("Answer: " + record["output"])
    return "\n".join(parts)


def run_sft(
    base_checkpoint: str,
    dataset_path: str,
    model_config: dict,
    training_config: dict,
    output: str,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer, SFTConfig

    os.makedirs(output, exist_ok=True)

    tokenizer_path = model_config.get("tokenizer_path") or base_checkpoint
    print(f"[finetune] Loading tokenizer from: {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[finetune] Loading model from: {base_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(base_checkpoint, torch_dtype=torch.float32)

    if len(tok) != model.config.vocab_size:
        model.resize_token_embeddings(len(tok))
        print(f"[finetune] Resized embeddings to {len(tok)}")

    lora_cfg = model_config.get("lora", {})
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["c_attn"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[finetune] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"[finetune] Loading dataset from: {dataset_path}")
    records = load_jsonl(dataset_path)
    split_filter = training_config.get("split", "train")
    if split_filter != "all":
        records = [r for r in records if r.get("split") == split_filter]
    print(f"[finetune] Records after split filter ({split_filter!r}): {len(records)}")

    texts = [format_record(r) for r in records]
    hf_dataset = Dataset.from_dict({"text": texts})

    epochs = training_config.get("num_epochs", 3)
    batch_size = training_config.get("batch_size", 4)
    lr = training_config.get("learning_rate", 2e-4)
    max_seq_len = training_config.get("max_seq_length", 512)
    warmup_steps = training_config.get("warmup_steps", 50)
    logging_steps = training_config.get("logging_steps", 10)

    sft_config = SFTConfig(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        max_length=max_seq_len,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_dataset,
        processing_class=tok,
    )

    print(f"[finetune] Starting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"[finetune] Training complete in {elapsed:.1f}s")
    trainer.save_model(output)
    tok.save_pretrained(output)

    final_loss = train_result.training_loss
    print(f"[finetune] Final training loss: {final_loss:.4f}")

    manifest = {
        "base_checkpoint": base_checkpoint,
        "dataset": dataset_path,
        "output": output,
        "split_filter": split_filter,
        "num_records": len(records),
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_seq_length": max_seq_len,
        "lora_rank": lora_cfg.get("rank", 8),
        "lora_alpha": lora_cfg.get("alpha", 16),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "final_training_loss": final_loss,
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(output, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[finetune] Manifest written to: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT fine-tuning.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_config = load_yaml(args.model_config)
    training_config = load_yaml(args.training_config)

    run_sft(
        base_checkpoint=args.base_checkpoint,
        dataset_path=args.dataset,
        model_config=model_config,
        training_config=training_config,
        output=args.output,
    )


if __name__ == "__main__":
    main()
