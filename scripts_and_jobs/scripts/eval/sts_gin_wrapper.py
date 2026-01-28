#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained GIN models on STS tasks.
"""
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import LayerGINEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise


class STSGINWrapper:
    """MTEB-compatible wrapper for GIN models trained on STS tasks."""

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        model_path: str,
        device_map: str = "auto",
    ):
        """
        Initialize STS GIN wrapper.

        Args:
            model_specs: Specifications for the base model
            model_path: Path to trained STS GIN model checkpoint
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

        # Load trained GIN encoder
        self.gnn_encoder = self._load_encoder_from_checkpoint()
        self.gnn_encoder.eval()

    def _load_encoder_from_checkpoint(self) -> LayerGINEncoder:
        """Load GIN encoder from PairCosineSimScore checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key")

        saved_args = checkpoint['args']

        # Get input dimension
        sample_texts = ["Sample text"]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        input_dim = layerwise[0].shape[1]

        # Reconstruct GIN encoder
        encoder = LayerGINEncoder(
            in_dim=input_dim,
            hidden_dim=saved_args['gin_hidden_dim'],
            num_layers=saved_args['gin_layers'],
            dropout=saved_args['dropout'],
            gin_mlp_layers=saved_args.get('gin_mlp_layers', 1),
            node_to_choose=saved_args.get('node_to_choose', 'mean'),
            graph_type=saved_args['graph_type'],  # Load graph_type from checkpoint
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

        print(f"✓ Loaded STS GIN model from {self.model_path}")
        print(f"  Hidden dim: {saved_args['gin_hidden_dim']}")
        print(f"  GIN layers: {saved_args['gin_layers']}")
        print(f"  Graph type: {saved_args.get('graph_type', 'unknown')}")
        print(f"  Linear mode: {saved_args.get('use_linear', False)}")

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
            np.ndarray: Sentence embeddings [N, hidden_dim]
        """
        from experiments.utils.model_definitions.gnn.gnn_datasets import SingleGraphDataset
        from torch_geometric.loader import DataLoader as PyGDataLoader

        # Extract layer-wise embeddings
        layerwise = compute_layerwise(
            self.base_wrapper,
            sentences,
            batch_size=batch_size,
            token_pooling_method="mean"
        )

        # Create graph dataset
        dataset = SingleGraphDataset(
            layerwise_list=layerwise,
            labels=[0] * len(sentences),  # Dummy labels
            graph_type=self.gnn_encoder.graph_type  # Use graph_type from encoder
        )

        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Encode through GNN
        embeddings = []
        self.gnn_encoder.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                emb = self.gnn_encoder(batch)  # [B, hidden_dim]
                embeddings.append(emb.cpu().numpy())

        # Clear GPU cache to prevent OOM across tasks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(embeddings, axis=0)
