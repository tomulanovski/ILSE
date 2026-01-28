#!/usr/bin/env python3
"""
MTEB-compatible wrapper for trained DWAtt models on classification tasks.

DWAtt (Depth-Wise Attention) from ElNokrashy et al. 2024:
"Depth-Wise Attention (DWAtt): A Layer Fusion Method for Data-Efficient Classification"
https://arxiv.org/abs/2209.15168
"""
import os
from typing import List, Optional, Union
try:
    from mteb.encoder_interface import PromptType
except ImportError:
    PromptType = None
import numpy as np
import torch
from pathlib import Path

from experiments.utils.model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications
from experiments.utils.model_definitions.gnn.gnn_models import DWAttEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise, LayerwiseTensorDataset


class DWAttWrapper:
    """
    MTEB-compatible wrapper for DWAtt models trained on layer-wise embeddings.

    Directly uses DWAttEncoder without classifier wrapper overhead.
    """

    def __init__(
        self,
        model_specs: TextModelSpecifications,
        device_map: str = "auto",
        model_path: Optional[str] = None,
    ):
        """
        Initialize the DWAtt wrapper.

        Args:
            model_specs: Specifications for the base pre-trained model
            device_map: Device mapping for the base model
            model_path: Path to pre-trained DWAtt model (if available)
        """
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the base pre-trained model wrapper
        self.base_wrapper = TextLayerwiseAutoModelWrapper(
            model_specs,
            device_map=device_map,
            evaluation_layer_idx=-1  # Ignored by compute_layerwise anyway
        )

        # DWAtt encoder (the actual model we use)
        self.dwatt_encoder = None
        self.model_path = model_path

        # Cache for embeddings
        self._embedding_dim = None

    def _extract_encoder_from_checkpoint(self, checkpoint: dict) -> DWAttEncoder:
        """
        Extract and reconstruct the DWAtt encoder from a checkpoint.
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

        # Create DWAtt encoder with paper-faithful defaults
        encoder = DWAttEncoder(
            num_layers=L,
            layer_dim=D,
            hidden_dim=saved_args.get('dwatt_hidden_dim', None),
            bottleneck_ratio=saved_args.get('dwatt_bottleneck_ratio', 0.5),
            pos_embed_dim=saved_args.get('dwatt_pos_embed_dim', 24),
            dropout=saved_args.get('dropout', 0.1)
        )

        # Extract encoder weights from classifier checkpoint
        state_dict = checkpoint['model_state_dict']

        # Filter out only encoder weights (remove 'linear.' prefix weights)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('enc.'):
                # Remove 'enc.' prefix to match DWAttEncoder parameter names
                encoder_key = key[4:]  # Remove 'enc.'
                encoder_state_dict[encoder_key] = value

        # Load weights into encoder
        encoder.load_state_dict(encoder_state_dict)

        return encoder.to(self.device)

    def load_model(self, model_path: str):
        """Load a pre-trained DWAtt model and extract the encoder."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading DWAtt encoder from {model_path}...")
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract encoder from classifier checkpoint
            self.dwatt_encoder = self._extract_encoder_from_checkpoint(checkpoint)
            self.dwatt_encoder.eval()

        except Exception as e:
            self.dwatt_encoder = None
            raise RuntimeError(f"Failed to load DWAtt encoder from {model_path}: {e}")

        saved_args = checkpoint.get('args', {})
        hidden_dim = saved_args.get('dwatt_hidden_dim')
        hidden_str = f"{hidden_dim}" if hidden_dim else "paper-faithful (layer_dim)"
        print(f"Successfully loaded DWAtt encoder from {model_path}")
        print(f"  Architecture:")
        print(f"    - Hidden dim: {hidden_str}")
        print(f"    - Bottleneck ratio: {saved_args.get('dwatt_bottleneck_ratio', 0.5)}")
        print(f"    - Pos embed dim: {saved_args.get('dwatt_pos_embed_dim', 24)}")
        print(f"    - Dropout: {saved_args.get('dropout', 0.1)}")
        val_acc = checkpoint.get('val_acc', 'unknown')
        if isinstance(val_acc, float):
            print(f"  Performance: val_acc={val_acc:.4f}")
        else:
            print(f"  Performance: val_acc={val_acc}")

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Encode sentences using the DWAtt model.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            **kwargs: Additional arguments (ignored)

        Returns:
            np.ndarray: Embeddings of shape (N, embedding_dim)
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]

        # Load model if not already loaded
        if self.dwatt_encoder is None:
            if self.model_path is None:
                raise RuntimeError("No model loaded. Call load_model() first or provide model_path during initialization.")
            self.load_model(self.model_path)

        # Extract layer-wise embeddings from base model
        layerwise_embeddings = compute_layerwise(
            self.base_wrapper,
            sentences,
            batch_size=batch_size,
            token_pooling_method="mean"
        )

        # Convert to tensor dataset
        dataset = LayerwiseTensorDataset(layerwise_embeddings, [0] * len(sentences))

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Encode with DWAtt encoder
        all_embeddings = []
        self.dwatt_encoder.eval()

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                embeddings = self.dwatt_encoder(batch_x)  # [B, out_dim]
                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        embeddings_array = np.vstack(all_embeddings)

        # Cache embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = embeddings_array.shape[1]

        return embeddings_array

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        if self._embedding_dim is None:
            # Trigger a dummy encoding to determine dimension
            _ = self.encode(["Sample text for dimension inference."])
        return self._embedding_dim
