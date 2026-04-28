#!/usr/bin/env python3
"""
Standalone LoRA trainer for STS (Semantic Textual Similarity) tasks.
Mirrors sts_gin_trainer.py but uses PEFT LoRA fine-tuning.

Key differences from classification LoRA trainer:
- Uses AutoModel (embeddings) not AutoModelForSequenceClassification
- Encodes sentence pairs → cosine similarity → MSE loss
- Metric: Spearman correlation
- task_type=FEATURE_EXTRACTION (not SEQ_CLS)
"""
import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
import numpy as np
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ..text_automodel_wrapper import get_model_path
from .gnn_datasets import load_task_data


class STSPairDataset(Dataset):
    """Dataset for STS sentence pairs with similarity scores."""
    def __init__(self, texts_a, texts_b, scores, tokenizer, max_length=2048):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        enc_a = self.tokenizer(
            self.texts_a[idx], max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        enc_b = self.tokenizer(
            self.texts_b[idx], max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate for variable-length sentence pairs."""
    def pad_tensors(tensors):
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.size(0)
            padded.append(torch.cat([t, torch.zeros(pad_len, dtype=t.dtype)]))
        return torch.stack(padded)

    return {
        "input_ids_a": pad_tensors([item["input_ids_a"] for item in batch]),
        "attention_mask_a": pad_tensors([item["attention_mask_a"] for item in batch]),
        "input_ids_b": pad_tensors([item["input_ids_b"] for item in batch]),
        "attention_mask_b": pad_tensors([item["attention_mask_b"] for item in batch]),
        "score": torch.stack([item["score"] for item in batch]),
    }


def mean_pooling(hidden_states, attention_mask):
    """Mean pooling over non-padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
    return sum_embeddings / sum_mask


def get_target_modules(model_name):
    """Get LoRA target modules based on model architecture."""
    if "pythia" in model_name.lower():
        return ["query_key_value", "dense"]
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        return ["q_proj", "v_proj"]


def _cos_to_score(cos, min_score, max_score):
    """Map cosine similarity [-1, 1] to score range [min_score, max_score]."""
    normalized = (cos + 1.0) * 0.5
    return min_score + normalized * (max_score - min_score)


def main():
    parser = argparse.ArgumentParser(description="LoRA Training for STS tasks")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="410m")
    # LoRA hyperparams
    parser.add_argument("--lora_r", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
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
    try:
        val_data = load_task_data(args.task, "validation")
        print(f"Using existing validation split for {args.task}")
    except Exception:
        raise KeyError(f"No validation data for STS task: {args.task}")

    min_score = train_data["min_score"]
    max_score = train_data["max_score"]
    print(f"Score range: [{min_score}, {max_score}]")

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
    base_model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=model_dtype, **extra_kwargs)

    # Apply LoRA
    target_modules = get_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
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
    train_dataset = STSPairDataset(
        train_data["text_a"], train_data["text_b"], train_data["original_scores"],
        tokenizer, max_length
    )
    val_dataset = STSPairDataset(
        val_data["text_a"], val_data["text_b"], val_data["original_scores"],
        tokenizer, max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "train_spearman": [], "val_loss": [], "val_spearman": [], "lr": []}
    best_val_spearman = -1.0
    best_epoch = 0
    early_stopping_patience = 15
    no_improvement_count = 0

    model_filename = f"lora_{args.task}_{args.model_family}_{args.model_size}_r{args.lora_r}"
    best_adapter_dir = save_dir / f"{model_filename}_adapter"
    best_checkpoint_path = save_dir / f"{model_filename}.pt"

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()
    print(f"\nStarting STS training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_total = 0
        all_train_preds = []
        all_train_scores = []

        for batch in train_loader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            scores = batch["score"].to(device)

            optimizer.zero_grad()

            out_a = model(input_ids=ids_a, attention_mask=mask_a)
            emb_a = mean_pooling(out_a.last_hidden_state, mask_a)

            out_b = model(input_ids=ids_b, attention_mask=mask_b)
            emb_b = mean_pooling(out_b.last_hidden_state, mask_b)

            cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
            score_preds = _cos_to_score(cos_sim, min_score, max_score)

            loss = criterion(score_preds, scores)
            loss.backward()
            optimizer.step()

            bs = scores.size(0)
            train_total += bs
            train_loss += loss.item() * bs
            all_train_preds.append(score_preds.detach().cpu())
            all_train_scores.append(scores.detach().cpu())

        train_preds_np = torch.cat(all_train_preds).numpy()
        train_scores_np = torch.cat(all_train_scores).numpy()
        train_spearman, _ = spearmanr(train_scores_np, train_preds_np)
        if not np.isfinite(train_spearman):
            train_spearman = 0.0
        avg_train_loss = train_loss / max(1, train_total)

        # Validate
        model.eval()
        val_loss = 0.0
        val_total = 0
        all_val_preds = []
        all_val_scores = []

        with torch.no_grad():
            for batch in val_loader:
                ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)
                scores = batch["score"].to(device)

                out_a = model(input_ids=ids_a, attention_mask=mask_a)
                emb_a = mean_pooling(out_a.last_hidden_state, mask_a)

                out_b = model(input_ids=ids_b, attention_mask=mask_b)
                emb_b = mean_pooling(out_b.last_hidden_state, mask_b)

                cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
                score_preds = _cos_to_score(cos_sim, min_score, max_score)
                loss = criterion(score_preds, scores)

                bs = scores.size(0)
                val_total += bs
                val_loss += loss.item() * bs
                all_val_preds.append(score_preds.detach().cpu())
                all_val_scores.append(scores.detach().cpu())

        val_preds_np = torch.cat(all_val_preds).numpy()
        val_scores_np = torch.cat(all_val_scores).numpy()
        val_spearman, _ = spearmanr(val_scores_np, val_preds_np)
        if not np.isfinite(val_spearman):
            val_spearman = 0.0
        avg_val_loss = val_loss / max(1, val_total)

        scheduler.step(val_spearman)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["train_spearman"].append(train_spearman)
        history["val_loss"].append(avg_val_loss)
        history["val_spearman"].append(val_spearman)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch+1:2d}: Train Loss {avg_train_loss:.4f} Spearman {train_spearman:.4f} | "
              f"Val Loss {avg_val_loss:.4f} Spearman {val_spearman:.4f} | LR {lr_now:.2e}")

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch + 1
            no_improvement_count = 0

            elapsed_time = time.time() - start_time
            current_peak_mem = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                current_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # Save PEFT adapter
            model.save_pretrained(str(best_adapter_dir))

            # Save metadata checkpoint
            torch.save({
                "epoch": epoch + 1,
                "train_spearman": train_spearman,
                "val_spearman": val_spearman,
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
            print(f"  New best model saved! Val Spearman: {val_spearman:.4f}")
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
    print(f"Done. Best Val Spearman: {best_val_spearman:.4f} @ epoch {best_epoch}")
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

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"Loss — {args.task} (LoRA r={args.lora_r})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{model_filename}_loss.png", dpi=160); plt.close()

    # Spearman plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_spearman"], label="Train Spearman")
    plt.plot(history["epoch"], history["val_spearman"], label="Val Spearman")
    plt.xlabel("Epoch"); plt.ylabel("Spearman Correlation")
    plt.title(f"Spearman — {args.task} (LoRA r={args.lora_r})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{model_filename}_spearman.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
