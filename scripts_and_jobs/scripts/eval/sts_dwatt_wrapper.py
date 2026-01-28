#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained DWAtt models on STS tasks.

DWAtt (Depth-Wise Attention) from ElNokrashy et al. 2024:
"Depth-Wise Attention (DWAtt): A Layer Fusion Method for Data-Efficient Classification"
https://arxiv.org/abs/2209.15168
"""
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import DWAttEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise


class STSDWAttWrapper:
    """MTEB-compatible wrapper for DWAtt models trained on STS tasks."""

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        model_path: str,
        device_map: str = "auto",
    ):
        """
        Initialize STS DWAtt wrapper.

        Args:
            model_specs: Specifications for the base model
            model_path: Path to trained STS DWAtt model checkpoint
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

        # Load trained DWAtt encoder
        self.dwatt_encoder = self._load_encoder_from_checkpoint()
        self.dwatt_encoder.eval()

    def _load_encoder_from_checkpoint(self) -> DWAttEncoder:
        """Load DWAtt encoder from PairCosineSimScore checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get dimensions from a sample forward pass
        sample_texts = ["Sample text"]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        num_layers, layer_dim = layerwise[0].shape

        # Extract hyperparameters from checkpoint args (saved via vars(args))
        # Default to paper-faithful values
        args = checkpoint.get('args', {})
        dwatt_hidden_dim = args.get('dwatt_hidden_dim', None)
        dwatt_bottleneck_ratio = args.get('dwatt_bottleneck_ratio', 0.5)
        dwatt_pos_embed_dim = args.get('dwatt_pos_embed_dim', 24)
        dropout = args.get('dropout', 0.1)

        # Reconstruct DWAtt encoder
        encoder = DWAttEncoder(
            num_layers=num_layers,
            layer_dim=layer_dim,
            hidden_dim=dwatt_hidden_dim,
            bottleneck_ratio=dwatt_bottleneck_ratio,
            pos_embed_dim=dwatt_pos_embed_dim,
            dropout=dropout
        ).to(self.device)

        # Load weights (extract encoder from PairCosineSimScore)
        state_dict = checkpoint['model_state_dict']
        encoder_state_dict = {
            k.replace('enc.', ''): v
            for k, v in state_dict.items()
            if k.startswith('enc.')
        }
        encoder.load_state_dict(encoder_state_dict)

        print(f"Loaded STS DWAtt model from {self.model_path}")
        print(f"  Num layers: {num_layers}")
        print(f"  Layer dim: {layer_dim}")
        print(f"  Hidden dim: {dwatt_hidden_dim if dwatt_hidden_dim else 'paper-faithful (layer_dim)'}")
        print(f"  Bottleneck ratio: {dwatt_bottleneck_ratio}")
        print(f"  Pos embed dim: {dwatt_pos_embed_dim}")
        print(f"  Output dim: {encoder.out_dim}")
        print(f"  Trainable params: {sum(p.numel() for p in encoder.parameters()):,}")

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
            np.ndarray: Sentence embeddings [N, out_dim]
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

        # Encode through DWAtt encoder
        embeddings = []
        self.dwatt_encoder.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]  # [B, L, D]
                emb = self.dwatt_encoder(batch)  # [B, out_dim]
                embeddings.append(emb.cpu().numpy())

        # Clear GPU cache to prevent OOM across tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)
