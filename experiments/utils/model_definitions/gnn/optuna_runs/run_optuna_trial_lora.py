import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from argparse import Namespace
from sklearn.model_selection import train_test_split
import numpy as np
import time
from tqdm import tqdm

from ..gnn_datasets import load_task_data
from ...text_automodel_wrapper import get_model_path

# Classification tasks supported
CLASSIFICATION_TASKS = [
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "MTOPIntentClassification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "MTOPDomainClassification",
    "MassiveScenarioClassification"
]

# Global variables set from command line
TASK_NAME = None
MODEL_FAMILY = None
MODEL_SIZE = None


class TextClassificationDataset(Dataset):
    """Dataset for text classification."""
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


def train_and_eval_lora(args: Namespace) -> dict:
    """
    Train and evaluate LoRA model for classification.

    NOTE: Only evaluates on validation set during Optuna trials (no test set).
    This matches GIN/MLP behavior for fair hyperparameter search comparison.
    Test set evaluation should be done separately after finding best hyperparameters.

    Returns dict with: best_val_acc, best_epoch, param_count, train_time_sec, epoch_logs
    """
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = load_task_data(args.task, "train")

    # Try to load validation data, if not available, split from train
    val_split_ratio = getattr(args, "val_split_ratio", 0.15)
    try:
        val_data = load_task_data(args.task, "validation")
        print(f"Using existing validation split for {args.task}")
    except Exception:
        print(f"No validation split found for {args.task}. Splitting {val_split_ratio*100:.0f}% from train data.")

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data["text"],
            train_data["labels"],
            test_size=val_split_ratio,
            random_state=seed,
            stratify=train_data["labels"]
        )

        train_data["text"] = train_texts
        train_data["labels"] = train_labels
        val_data = {
            "text": val_texts,
            "labels": val_labels,
            "num_classes": train_data["num_classes"],
        }
        print(f"Split created: {len(train_texts)} train samples, {len(val_texts)} validation samples")

    num_classes = train_data["num_classes"]

    # Load model and tokenizer
    model_name = get_model_path(args.model_family, args.model_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Match tokenization settings from GIN/MLP (text_automodel_wrapper.py)
    max_length = 512 if args.model_family.lower() in ["bert", "roberta"] else 2048

    # Load base model in bfloat16/float16 to match GIN/MLP/Weighted baselines
    # (TextLayerwiseAutoModelWrapper also loads in bfloat16/float16)
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Gemma2 produces NaN with SDPA attention — must use eager
    extra_kwargs = {"attn_implementation": "eager"} if "gemma" in model_name.lower() else {}
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        device_map="auto",
        torch_dtype=model_dtype,
        **extra_kwargs,
    )

    # Ensure pad_token_id is set
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA based on model architecture
    if "pythia" in model_name.lower():
        target_modules = ["query_key_value", "dense"]
    elif "llama" in model_name.lower() or "gemma" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Create datasets
    train_dataset = TextClassificationDataset(
        train_data["text"], train_data["labels"], tokenizer, max_length=max_length
    )
    val_dataset = TextClassificationDataset(
        val_data["text"], val_data["labels"], tokenizer, max_length=max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer and scheduler (loss computed by AutoModelForSequenceClassification)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    best_epoch = -1
    epoch_logs = []
    early_stopping_patience = getattr(args, "early_stopping_patience", 15)
    min_delta = getattr(args, "min_delta", 0.001)
    no_improvement_count = 0

    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()
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
                loss = outputs.loss

                val_loss += loss.item()
                predictions = outputs.logits.argmax(dim=-1)
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()

        val_acc = val_correct / val_total

        # Optuna intermediate reporting (for pruning)
        if hasattr(args, "trial"):
            args.trial.report(val_acc, step=epoch)
            if args.trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        scheduler.step(val_acc)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader)
        })

        # Early stopping
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            break

    duration = time.time() - start_time

    # Track peak GPU memory
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "param_count": trainable_params,  # Only trainable LoRA params
        "train_time_sec": duration,
        "peak_memory_mb": peak_memory_mb,
        "epoch_logs": epoch_logs
    }


def objective(trial):
    # Validate task is classification
    if TASK_NAME not in CLASSIFICATION_TASKS:
        raise ValueError(f"Task '{TASK_NAME}' is not a supported classification task. Supported: {CLASSIFICATION_TASKS}")

    args = Namespace(
        task=TASK_NAME,
        model_family=MODEL_FAMILY,
        model_size=MODEL_SIZE,
        # LoRA hyperparameters - search over r to control parameter count
        # r=1 (~196K params), r=2 (~393K params), r=3 (~590K params)
        lora_r=trial.suggest_categorical("lora_r", [1, 2, 3]),
        lora_alpha=trial.suggest_categorical("lora_alpha", [8, 16, 32]),
        lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.2, step=0.1),
        # Training hyperparameters
        epochs=20,  # LoRA typically needs fewer epochs than GIN
        batch_size=32,
        lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
        weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        save_dir="./lora_optuna",
        seed=trial.number,
        trial=trial
    )

    result = train_and_eval_lora(args)

    # Log custom metrics
    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])
    trial.set_user_attr("peak_memory_mb", result["peak_memory_mb"])
    trial.set_user_attr("epoch_logs", result["epoch_logs"])

    return result["best_val_acc"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for LoRA baseline on classification tasks")
    parser.add_argument("--study_name", type=str, required=True, help="Name for the Optuna study")
    parser.add_argument("--task", type=str, required=True, help="MTEB classification task name")
    parser.add_argument("--model_family", type=str, default="Pythia", help="Model family (e.g., Pythia, Llama3)")
    parser.add_argument("--model_size", type=str, default="410m", help="Model size (e.g., 410m, 2.8b, 8B)")
    parser.add_argument("--storage_url", type=str, required=True, help="PostgreSQL storage URL for Optuna")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of Optuna trials to run")
    args = parser.parse_args()

    # Set global variables for objective function
    TASK_NAME = args.task
    MODEL_FAMILY = args.model_family
    MODEL_SIZE = args.model_size

    print(f"Starting Optuna study: {args.study_name}")
    print(f"Task: {TASK_NAME}")
    print(f"Base Model: {MODEL_FAMILY}-{MODEL_SIZE}")
    print(f"Number of trials: {args.n_trials}")

    # Create pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,  # LoRA converges faster, check earlier
        interval_steps=3
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
        pruner=pruner
    )

    print(f"Pruner enabled: MedianPruner (stops trials with val_acc below median)")
    study.optimize(objective, n_trials=args.n_trials)
