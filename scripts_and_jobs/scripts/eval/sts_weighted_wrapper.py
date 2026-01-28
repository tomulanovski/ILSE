#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained Weighted models on STS tasks.
"""
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import LearnedWeightingEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise


class STSWeightedWrapper:
    """MTEB-compatible wrapper for Weighted models trained on STS tasks."""

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        model_path: str,
        device_map: str = "auto",
    ):
        """
        Initialize STS Weighted wrapper.

        Args:
            model_specs: Specifications for the base model
            model_path: Path to trained STS Weighted model checkpoint
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

        # Load trained Weighted encoder
        self.weighted_encoder = self._load_encoder_from_checkpoint()
        self.weighted_encoder.eval()

    def _load_encoder_from_checkpoint(self) -> LearnedWeightingEncoder:
        """Load Weighted encoder from PairCosineSimScore checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get dimensions
        sample_texts = ["Sample text"]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        num_layers, layer_dim = layerwise[0].shape

        # Reconstruct Weighted encoder
        encoder = LearnedWeightingEncoder(
            num_layers=num_layers,
            layer_dim=layer_dim
        ).to(self.device)

        # Load weights (extract encoder from PairCosineSimScore)
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {
            k.replace('enc.', ''): v
            for k, v in state_dict.items()
            if k.startswith('enc.')
        }
        encoder.load_state_dict(encoder_state_dict)

        print(f"✓ Loaded STS Weighted model from {self.model_path}")
        print(f"  Num layers: {num_layers}")
        print(f"  Layer dim: {layer_dim}")
        print(f"  Trainable params: {sum(p.numel() for p in encoder.parameters())}")

        # Print learned weights
        weights = torch.softmax(encoder.layer_weights, dim=0).detach().cpu().numpy()
        weights = np.atleast_1d(weights)  # Ensure it's a proper 1-d array
        print(f"  Learned layer weights (top 5):")
        num_to_show = min(5, len(weights))
        top_indices = np.argsort(weights)[-num_to_show:][::-1]
        for idx in top_indices:
            print(f"    Layer {idx}: {weights[idx]:.4f}")

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
            np.ndarray: Sentence embeddings [N, layer_dim]
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

        # Encode through Weighted encoder
        embeddings = []
        self.weighted_encoder.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]  # [B, L, D]
                emb = self.weighted_encoder(batch)  # [B, D]
                embeddings.append(emb.cpu().numpy())

        # Clear GPU cache to prevent OOM across tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)
