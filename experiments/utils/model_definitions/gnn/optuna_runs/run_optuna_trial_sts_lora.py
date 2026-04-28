#!/usr/bin/env python3
"""
Optuna hyperparameter search for LoRA on STS (Semantic Textual Similarity) tasks.

Key difference from classification LoRA:
- Uses AutoModel (not AutoModelForSequenceClassification)
- Encodes sentence pairs → cosine similarity → MSE loss against gold scores
- Metric: Spearman correlation (not accuracy)
- Matches the STS GIN/MLP/Weighted pipeline pattern
"""
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from argparse import Namespace
from sklearn.model_selection import train_test_split
import numpy as np
import time
from scipy.stats import spearmanr

from ..gnn_datasets import load_task_data
from ...text_automodel_wrapper import get_model_path
from experiments.utils.gpu_tracking import GPUTracker

# STS tasks supported
STS_TASKS = [
    "STSBenchmark",
    "SICK-R",
    "BIOSSES",
]

# Global variables set from command line
TASK_NAME = None
MODEL_FAMILY = None
MODEL_SIZE = None


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


def train_and_eval_sts_lora(args):
    """
    Train and evaluate LoRA for STS tasks.
    Encodes sentence pairs, computes cosine similarity, MSE loss against gold scores.
    """
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = load_task_data(args.task, "train")
    try:
        val_data = load_task_data(args.task, "validation")
        print(f"Using existing validation split for {args.task}")
    except Exception:
        raise KeyError(f"No validation data for STS task: {args.task}")

    # Score range for mapping cosine similarity to scores
    min_score = train_data["min_score"]
    max_score = train_data["max_score"]
    print(f"Score range: [{min_score}, {max_score}]")

    # Load model and tokenizer
    model_name = get_model_path(args.model_family, args.model_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = 512 if args.model_family.lower() in ["bert", "roberta"] else 2048

    # Load base model (AutoModel for embeddings, not ForSequenceClassification)
    # Use bfloat16/float16 to match the precision used by GIN/MLP/Weighted baselines
    # (TextLayerwiseAutoModelWrapper also loads in bfloat16/float16)
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
        tokenizer, max_length=max_length
    )
    val_dataset = STSPairDataset(
        val_data["text_a"], val_data["text_b"], val_data["original_scores"],
        tokenizer, max_length=max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training loop
    best_val_spearman = -1.0
    best_epoch = -1
    epoch_logs = []
    early_stopping_patience = getattr(args, "early_stopping_patience", 15)
    no_improvement_count = 0

    # GPU tracking
    gpu_tracker = GPUTracker(device, require_gpu=False)
    gpu_tracker.start()

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

            # Encode both sentences
            out_a = model(input_ids=ids_a, attention_mask=mask_a)
            emb_a = mean_pooling(out_a.last_hidden_state, mask_a)

            out_b = model(input_ids=ids_b, attention_mask=mask_b)
            emb_b = mean_pooling(out_b.last_hidden_state, mask_b)

            # Cosine similarity → score prediction
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

        # Optuna pruning
        if hasattr(args, "trial"):
            args.trial.report(val_spearman, step=epoch)
            if args.trial.should_prune():
                raise optuna.TrialPruned()

        scheduler.step(val_spearman)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_spearman": train_spearman,
            "val_spearman": val_spearman,
            "train_loss": train_loss / max(1, train_total),
            "val_loss": val_loss / max(1, val_total),
        })

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    metrics = gpu_tracker.stop()

    return {
        "best_val_spearman": best_val_spearman,
        "best_epoch": best_epoch,
        "param_count": trainable_params,
        "train_time_sec": metrics["runtime_sec"],
        "peak_memory_mb": metrics["peak_memory_mb"],
        "peak_memory_gb": metrics["peak_memory_gb"],
        "epoch_logs": epoch_logs,
    }


def objective(trial):
    args = Namespace(
        task=TASK_NAME,
        model_family=MODEL_FAMILY,
        model_size=MODEL_SIZE,
        # LoRA hyperparameters
        lora_r=trial.suggest_categorical("lora_r", [1, 2, 3]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [8, 16, 32]),
        lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.2, step=0.1),
        # Training hyperparameters
        epochs=20,
        batch_size=32,
        lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
        weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        seed=trial.number,
        trial=trial,
    )

    result = train_and_eval_sts_lora(args)

    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])
    trial.set_user_attr("peak_memory_mb", result["peak_memory_mb"])
    trial.set_user_attr("epoch_logs", result["epoch_logs"])

    return result["best_val_spearman"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for LoRA on STS tasks")
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="STS task name (e.g., STSBenchmark, SICK-R)")
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="410m")
    parser.add_argument("--storage_url", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=1)
    args = parser.parse_args()

    TASK_NAME = args.task
    MODEL_FAMILY = args.model_family
    MODEL_SIZE = args.model_size

    print(f"Starting STS LoRA Optuna study: {args.study_name}")
    print(f"Task: {TASK_NAME}")
    print(f"Base Model: {MODEL_FAMILY}-{MODEL_SIZE}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=3,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=args.n_trials)
