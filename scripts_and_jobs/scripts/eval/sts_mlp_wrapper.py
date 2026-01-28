#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained MLP models on STS tasks.
"""
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import MLPEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise


class STSMLPWrapper:
    """MTEB-compatible wrapper for MLP models trained on STS tasks."""

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        model_path: str,
        device_map: str = "auto",
    ):
        """
        Initialize STS MLP wrapper.

        Args:
            model_specs: Specifications for the base model
            model_path: Path to trained STS MLP model checkpoint
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

        # Load trained MLP encoder
        self.mlp_encoder, self.mlp_input_mode = self._load_encoder_from_checkpoint()
        self.mlp_encoder.eval()

    def _load_encoder_from_checkpoint(self):
        """Load MLP encoder from PairCosineSimScore checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key")

        saved_args = checkpoint['args']
        mlp_input_mode = saved_args.get('mlp_input', 'last')

        # Get input dimension
        sample_texts = ["Sample text"]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        L, D = layerwise[0].shape

        if mlp_input_mode in ("last", "mean"):
            in_dim = D
        elif mlp_input_mode == "flatten":
            in_dim = L * D
        else:
            raise ValueError(f"Unknown mlp_input mode: {mlp_input_mode}")

        # Reconstruct MLP encoder
        encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=saved_args.get('mlp_hidden_dim', 256),
            num_layers=saved_args.get('mlp_layers', 2),
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

        print(f"✓ Loaded STS MLP model from {self.model_path}")
        print(f"  MLP input: {mlp_input_mode}")
        print(f"  Hidden dim: {saved_args.get('mlp_hidden_dim', 256)}")
        print(f"  MLP layers: {saved_args.get('mlp_layers', 2)}")
        print(f"  Linear mode: {saved_args.get('use_linear', False)}")

        return encoder, mlp_input_mode

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
            np.ndarray: Sentence embeddings [N, hidden_dim]
        """
        # Extract layer-wise embeddings
        layerwise = compute_layerwise(
            self.base_wrapper,
            sentences,
            batch_size=batch_size,
            token_pooling_method="mean"
        )

        # Process according to mlp_input mode
        if self.mlp_input_mode == "last":
            X = np.array([lw[-1, :] for lw in layerwise])  # [N, D]
        elif self.mlp_input_mode == "mean":
            X = np.array([lw.mean(axis=0) for lw in layerwise])  # [N, D]
        elif self.mlp_input_mode == "flatten":
            X = np.array([lw.flatten() for lw in layerwise])  # [N, L*D]
        else:
            raise ValueError(f"Unknown mlp_input mode: {self.mlp_input_mode}")

        # Convert to tensor and encode
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        embeddings = []
        self.mlp_encoder.eval()
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                emb = self.mlp_encoder(batch)  # [B, hidden_dim]
                embeddings.append(emb.cpu().numpy())

        # Clear GPU cache to prevent OOM across tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)
