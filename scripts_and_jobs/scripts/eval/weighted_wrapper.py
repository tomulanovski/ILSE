#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained Weighted models.
Directly uses LearnedWeightingEncoder without unnecessary classifier wrapper.
"""
import os
from typing import List, Optional, Dict, Any, Union
try:
    from mteb.encoder_interface import PromptType
except ImportError:
    # Fallback for older MTEB versions
    PromptType = None
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import LearnedWeightingEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise, LayerwiseTensorDataset


class WeightedWrapper:
    """
    MTEB-compatible wrapper for Weighted models trained on layer-wise embeddings.

    Directly uses LearnedWeightingEncoder without classifier wrapper overhead.
    """

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        device_map: str = "auto",
        model_path: Optional[str] = None,
    ):
        """
        Initialize the Weighted wrapper.

        Args:
            model_specs: Specifications for the base pre-trained model
            device_map: Device mapping for the base model
            model_path: Path to pre-trained Weighted model (if available)
        """
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the base pre-trained model wrapper
        self.base_wrapper = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=device_map,
            evaluation_layer_idx=-1  # Ignored by compute_layerwise anyway
        )

        # Weighted encoder (the actual model we use)
        self.weighted_encoder = None
        self.model_path = model_path

        # Cache for embeddings
        self._embedding_dim = None

    def _extract_encoder_from_checkpoint(self, checkpoint: dict) -> LearnedWeightingEncoder:
        """
        Extract and reconstruct the Weighted encoder from a checkpoint.
        """
        # Get training hyperparameters
        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key. Cannot reconstruct model architecture.")

        saved_args = checkpoint['args']

        # Get input dimension by running sample text through base model
        sample_texts = ["Sample text for dimension inference."]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)

        # Determine dimensions from layerwise embeddings
        L, D = layerwise[0].shape  # L = num_layers, D = layer_dim

        # Create Weighted encoder
        encoder = LearnedWeightingEncoder(
            num_layers=L,
            layer_dim=D
        )

        # Extract encoder weights from classifier checkpoint
        state_dict = checkpoint['model_state_dict']

        # Filter out only encoder weights (remove 'linear.' prefix weights)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('enc.'):
                # Remove 'enc.' prefix to match LearnedWeightingEncoder parameter names
                encoder_key = key[4:]  # Remove 'enc.'
                encoder_state_dict[encoder_key] = value

        # Load weights into encoder
        encoder.load_state_dict(encoder_state_dict)

        return encoder.to(self.device)

    def load_model(self, model_path: str):
        """Load a pre-trained Weighted model and extract the encoder."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading Weighted encoder from {model_path}...")
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract encoder from classifier checkpoint
            self.weighted_encoder = self._extract_encoder_from_checkpoint(checkpoint)
            self.weighted_encoder.eval()

        except Exception as e:
            self.weighted_encoder = None
            raise RuntimeError(f"Failed to load Weighted encoder from {model_path}: {e}")

        print(f"✓ Successfully loaded Weighted encoder from {model_path}")
        print(f"  Architecture: {self.weighted_encoder.num_layers} layers, softmax weighting")
        print(f"  Performance: val_acc={checkpoint.get('val_acc', 'unknown'):.4f}" if isinstance(checkpoint.get('val_acc'), float) else f"  Performance: val_acc={checkpoint.get('val_acc', 'unknown')}")

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
        Encode sentences using the Weighted encoder directly.

        Args:
            sentences: Input sentences
            task_name: Name of the MTEB task (ignored)
            prompt_type: Prompt type for task (ignored)
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings [N, layer_dim]
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # Step 1: Get layerwise embeddings from base model
        layerwise_embeddings = compute_layerwise(
            self.base_wrapper,
            sentences,
            batch_size=batch_size,
            token_pooling_method="mean"
        )

        # Safety check
        if self.weighted_encoder is None:
            raise RuntimeError(
                "Weighted encoder is not loaded! Call load_model() with a valid model path before encoding."
            )

        # Step 2: Create layerwise tensor dataset
        dummy_labels = [0] * len(layerwise_embeddings)
        dataset = LayerwiseTensorDataset(
            layerwise_embeddings,
            dummy_labels
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Step 3: Run tensors through Weighted encoder directly
        embeddings_list = []
        self.weighted_encoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch  # y is dummy labels, ignore
                X = X.to(self.device)
                # Direct call to Weighted encoder - no classifier wrapper!
                weighted_embeddings = self.weighted_encoder(X)  # [B, layer_dim]
                embeddings_list.append(weighted_embeddings.cpu().numpy())

        # Step 4: Concatenate and optionally normalize
        embeddings = np.concatenate(embeddings_list, axis=0)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            if self.weighted_encoder is not None:
                self._embedding_dim = self.weighted_encoder.layer_dim
            else:
                # Cannot determine without loading model
                raise RuntimeError("Cannot determine embedding dimension without loading a model first.")
        return self._embedding_dim

    def __repr__(self):
        return f"WeightedWrapper(model={self.model_specs.model_family}-{self.model_specs.model_size}, num_layers={self.weighted_encoder.num_layers if self.weighted_encoder else 'unknown'})"
