#!/usr/bin/env python3
"""
MTEB-compatible wrapper for LoRA fine-tuned models.
Loads base LLM + PEFT adapter and extracts last hidden state embeddings.

Unlike GIN/MLP/Weighted wrappers which extract layerwise embeddings then aggregate,
this wrapper uses the LoRA-modified model directly to produce improved hidden states.
"""
import os
from typing import List, Optional, Union
try:
    from mteb.encoder_interface import PromptType
except ImportError:
    PromptType = None
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications, get_model_path


class LoRAWrapper:
    """
    MTEB-compatible wrapper for LoRA fine-tuned models.

    Loads base model + LoRA adapter, extracts mean-pooled last hidden state
    as the embedding (same approach as standard sentence transformers).
    """

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        device_map: str = "auto",
        model_path: Optional[str] = None,
    ):
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.model_path = model_path

        self.model = None
        self.tokenizer = None
        self._embedding_dim = None

    def load_model(self, model_path: str):
        """
        Load base model + LoRA adapter.

        Args:
            model_path: Path to .pt checkpoint (must contain 'adapter_dir' key)
                        or directly to the adapter directory.
        """
        model_name = get_model_path(self.model_specs.model_family, self.model_specs.model_size)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine adapter directory
        if model_path.endswith(".pt"):
            checkpoint = torch.load(model_path, map_location="cpu")
            adapter_dir = checkpoint.get("adapter_dir", None)
            if adapter_dir is None:
                # Try conventional path: same name but _adapter suffix
                adapter_dir = model_path.replace(".pt", "_adapter")
            print(f"Loading LoRA adapter from: {adapter_dir}")
            if not os.path.exists(adapter_dir):
                raise FileNotFoundError(
                    f"Adapter directory not found: {adapter_dir}\n"
                    f"Expected alongside checkpoint: {model_path}"
                )
        else:
            adapter_dir = model_path

        # Detect task type from adapter config to use matching base model class
        # Classification LoRA: trained with AutoModelForSequenceClassification (SEQ_CLS)
        # STS LoRA: trained with AutoModel (FEATURE_EXTRACTION)
        adapter_config_path = Path(adapter_dir) / "adapter_config.json"
        is_seq_cls = True  # default: classification
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            task_type = adapter_cfg.get("task_type", "SEQ_CLS")
            is_seq_cls = (task_type == "SEQ_CLS")
            print(f"  Adapter task_type: {task_type}")

        # Use bfloat16 to match training precision (avoids NaN on Gemma2 and saves memory)
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Gemma2 produces NaN with SDPA attention — must use eager
        extra_kwargs = {}
        if "gemma" in model_name.lower():
            extra_kwargs["attn_implementation"] = "eager"

        print(f"Loading base model: {model_name} (dtype={model_dtype})")
        if is_seq_cls:
            # Classification: must use AutoModelForSequenceClassification to match module paths
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, device_map=self.device_map,
                torch_dtype=model_dtype, **extra_kwargs
            )
            if base_model.config.pad_token_id is None:
                base_model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            # STS/Feature extraction: use AutoModel
            base_model = AutoModel.from_pretrained(
                model_name, device_map=self.device_map,
                torch_dtype=model_dtype, **extra_kwargs
            )

        # Load LoRA adapter on top
        print(f"Applying LoRA adapter from: {adapter_dir}")
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        self._is_seq_cls = is_seq_cls

        # Get embedding dimension from config
        self._embedding_dim = base_model.config.hidden_size

        print(f"  LoRA model loaded successfully")
        print(f"  Embedding dim: {self._embedding_dim}")
        if model_path.endswith(".pt"):
            if 'val_acc' in checkpoint:
                print(f"  Training val_acc: {checkpoint['val_acc']:.4f}")
            elif 'val_spearman' in checkpoint:
                print(f"  Training val_spearman: {checkpoint['val_spearman']:.4f}")

    def encode(
        self,
        sentences: Union[str, List[str]],
        task_name: Optional[str] = None,
        prompt_type: Optional["PromptType"] = None,
        batch_size: Optional[int] = 32,
        normalize_embeddings: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences using the LoRA fine-tuned model.

        Extracts mean-pooled last hidden state as embedding.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_length = 512 if self.model_specs.model_family.lower() in ["bert", "roberta"] else 2048

        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i:i + batch_size]

            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                if self._is_seq_cls:
                    # Classification model: extract hidden states before classification head
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
                else:
                    # Feature extraction model: last_hidden_state is directly available
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

                # Mean pooling over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
                embeddings = sum_embeddings / sum_mask  # [batch, hidden_dim]

            all_embeddings.append(embeddings.float().cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)

        return embeddings

    def get_embedding_dimension(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError("Cannot determine embedding dimension without loading a model first.")
        return self._embedding_dim

    def __repr__(self):
        return (f"LoRAWrapper(model={self.model_specs.model_family}-{self.model_specs.model_size}, "
                f"dim={self._embedding_dim or 'unknown'})")
