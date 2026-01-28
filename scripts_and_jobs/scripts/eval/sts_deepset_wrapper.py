#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained DeepSet models on STS tasks.
"""
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import DeepSetEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise


class STSDeepSetWrapper:
    """MTEB-compatible wrapper for DeepSet models trained on STS tasks."""

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        model_path: str,
        device_map: str = "auto",
    ):
        """
        Initialize STS DeepSet wrapper.

        Args:
            model_specs: Specifications for the base model
            model_path: Path to trained STS DeepSet model checkpoint
            device_map: Device mapping for base model
        """
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Initialize base model wrapper
        self.base_wrapper = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=device_map,
            evaluation_layer_idx=-1
        )

        # Load trained DeepSet encoder
        self.deepset_encoder = self._load_encoder_from_checkpoint()
        self.deepset_encoder.eval()

    def _load_encoder_from_checkpoint(self) -> DeepSetEncoder:
        """Load DeepSet encoder from PairCosineSimScore checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key")

        saved_args = checkpoint['args']

        # Get dimensions
        sample_texts = ["Sample text"]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        num_layers, layer_dim = layerwise[0].shape

        # Reconstruct DeepSet encoder
        encoder = DeepSetEncoder(
            num_layers=num_layers,
            layer_dim=layer_dim,
            hidden_dim=saved_args.get('deepset_hidden_dim', 256),
            pre_pooling_layers=saved_args.get('deepset_pre_pooling_layers', 0),
            post_pooling_layers=saved_args.get('deepset_post_pooling_layers', 1),
            pooling_type=saved_args.get('deepset_pooling_type', 'mean'),
            dropout=saved_args.get('dropout', 0.1),
            use_linear=saved_args.get('use_linear', False)  # Load use_linear flag from checkpoint
        ).to(self.device)

        # Load weights (extract encoder from PairCosineSimScore)
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {
            k.replace('enc.', ''): v
            for k, v in state_dict.items()
            if k.startswith('enc.')
        }
        encoder.load_state_dict(encoder_state_dict)

        print(f"✓ Loaded STS DeepSet model from {self.model_path}")
        print(f"  Num layers: {num_layers}")
        print(f"  Layer dim: {layer_dim}")
        print(f"  Hidden dim: {saved_args.get('deepset_hidden_dim', 256)}")
        print(f"  Pre-pooling layers: {saved_args.get('deepset_pre_pooling_layers', 0)}")
        print(f"  Post-pooling layers: {saved_args.get('deepset_post_pooling_layers', 1)}")
        print(f"  Pooling type: {saved_args.get('deepset_pooling_type', 'mean')}")
        print(f"  Linear mode: {saved_args.get('use_linear', False)}")
        print(f"  Trainable params: {sum(p.numel() for p in encoder.parameters())}")

        return encoder

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences to embeddings.

        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for processing

        Returns:
            np.ndarray: Sentence embeddings [N, hidden_dim or layer_dim]
        """
        # Extract layer-wise embeddings
        layerwise = compute_layerwise(
            self.base_wrapper,
            sentences,
            batch_size=batch_size,
            token_pooling_method="mean"
        )

        # Convert to tensor [N, L, D]
        X = torch.tensor(np.array(layerwise), dtype=torch.float32).to(self.device)

        # Encode through DeepSet encoder
        embeddings = []
        self.deepset_encoder.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]  # [B, L, D]
                emb = self.deepset_encoder(batch)  # [B, hidden_dim or D]
                embeddings.append(emb.cpu().numpy())

        # Clear GPU cache to prevent OOM across tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)
