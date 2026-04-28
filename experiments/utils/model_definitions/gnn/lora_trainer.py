#!/usr/bin/env python3
"""
Standalone LoRA trainer for training best models from Optuna results.
Mirrors the interface of basic_gin_trainer.py but uses PEFT LoRA fine-tuning.

Key differences from GIN/MLP/Weighted trainers:
- Loads base LLM and applies LoRA adapters (not extract-then-train)
- Uses AutoModelForSequenceClassification (classification head built-in)
- Saves PEFT adapter weights + metadata (not full model)
- Requires base model in GPU memory throughout training
"""
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import numpy as np
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..text_automodel_wrapper import get_model_path
from .gnn_datasets import load_task_data


class TextClassificationDataset(Dataset):
    """Dataset for text classification with tokenization."""
    def __init__(self, texts, labels, tokenizer, max_length=2048):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate to handle variable-length sequences with dynamic padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # Pad to max length in this batch
    max_len = max(ids.size(0) for ids in input_ids)
    padded_ids = []
    padded_masks = []
    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - ids.size(0)
        padded_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
        padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
        "labels": labels,
    }


def get_target_modules(model_name):
    """Get LoRA target modules based on model architecture."""
    if "pythia" in model_name.lower():
        return ["query_key_value", "dense"]
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        return ["q_proj", "v_proj"]


def main():
    parser = argparse.ArgumentParser(description="LoRA Training on base LLM for classification")
    # Task and model
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="410m")
    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=2, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # Other
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Task: {args.task} | LoRA r={args.lora_r}")

    # Load data
    train_data = load_task_data(args.task, "train")

    val_split_ratio = 0.15
    try:
        val_data = load_task_data(args.task, "validation")
        print(f"Using existing validation split for {args.task}")
    except Exception:
        print(f"No validation split found. Splitting {val_split_ratio*100:.0f}% from train.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data["text"], train_data["labels"],
            test_size=val_split_ratio, random_state=args.seed,
            stratify=train_data["labels"]
        )
        train_data["text"] = train_texts
        train_data["labels"] = train_labels
        val_data = {"text": val_texts, "labels": val_labels, "num_classes": train_data["num_classes"]}

    num_classes = train_data["num_classes"]

    # Load model and tokenizer
    model_name = get_model_path(args.model_family, args.model_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = 512 if args.model_family.lower() in ["bert", "roberta"] else 2048

    # Load base model in bfloat16/float16 to match GIN/MLP/Weighted baselines
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Gemma2 produces NaN with SDPA attention — must use eager
    extra_kwargs = {"attn_implementation": "eager"} if "gemma" in model_name.lower() else {}
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, device_map="auto", torch_dtype=model_dtype, **extra_kwargs,
    )
    # Ensure pad_token_id is set
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    target_modules = get_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Create datasets
    train_dataset = TextClassificationDataset(train_data["text"], train_data["labels"], tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_data["text"], val_data["labels"], tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer and scheduler (loss computed by AutoModelForSequenceClassification)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_patience = 15
    min_delta = 0.001
    no_improvement_count = 0

    # Model save path: lora_{task}_{model_family}_{model_size}_r{lora_r}.pt
    model_filename = f"lora_{args.task}_{args.model_family}_{args.model_size}_r{args.lora_r}"
    best_adapter_dir = save_dir / f"{model_filename}_adapter"
    best_checkpoint_path = save_dir / f"{model_filename}.pt"

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()
    print(f"\nStarting training for {args.epochs} epochs (early stopping patience={early_stopping_patience})...")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            train_total += labels.size(0)
            train_correct += (predictions == labels).sum().item()

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch+1:2d}: Train Loss {avg_train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {avg_val_loss:.4f} Acc {val_acc:.4f} | LR {lr_now:.2e}")

        # Check for improvement
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improvement_count = 0

            elapsed_time = time.time() - start_time
            current_peak_mem = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                current_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # Save PEFT adapter
            model.save_pretrained(str(best_adapter_dir))

            # Also save a metadata checkpoint (for pipeline compatibility)
            torch.save({
                "epoch": epoch + 1,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_time_sec": elapsed_time,
                "peak_memory_mb": current_peak_mem,
                "param_count": trainable_params,
                "args": vars(args),
                "adapter_dir": str(best_adapter_dir),
                "lora_config": {
                    "r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "target_modules": target_modules,
                },
            }, best_checkpoint_path)
            print(f"  New best model saved! Val Acc: {val_acc:.4f}")
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"\n  Early stopping: no improvement for {early_stopping_patience} epochs")
            break

    # Final metrics
    train_time_sec = time.time() - start_time
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print("-" * 60)
    print(f"Done. Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Training time: {train_time_sec:.1f}s | Peak GPU memory: {peak_memory_mb:.1f} MB")
    print(f"Trainable params: {trainable_params:,} (LoRA r={args.lora_r})")
    print(f"Saved adapter: {best_adapter_dir}")
    print(f"Saved checkpoint: {best_checkpoint_path}")

    # Save history CSV
    hist_path = save_dir / f"{model_filename}_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(history.keys())
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"Saved history: {hist_path}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss — {args.task} (LoRA r={args.lora_r})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{model_filename}_loss.png", dpi=160); plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy — {args.task} (LoRA r={args.lora_r})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{model_filename}_acc.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
