#!/usr/bin/env python3
"""
 MTEB-compatible wrapper for trained MLP models.
Directly uses MLPEncoder without unnecessary classifier wrapper.
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
from experiments.utils.model_definitions.gnn.gnn_models import MLPEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise, SimpleTensorDataset


class MLPWrapper:
    """
     MTEB-compatible wrapper for MLP models trained on layer-wise embeddings.
    
    Directly uses MLPEncoder without classifier wrapper overhead.
    """
    
    def __init__(
        self,
        model_specs: TextModelSpecifications,
        device_map: str = "auto",
        model_path: Optional[str] = None,
    ):
        """
        Initialize the MLP wrapper.
        
        Args:
            model_specs: Specifications for the base pre-trained model
            device_map: Device mapping for the base model
            model_path: Path to pre-trained MLP model (if available)
        """
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the base pre-trained model wrapper
        self.base_wrapper = TextLayerwiseAutoModelWrapper(
            model_specs, 
            device_map=device_map,
            evaluation_layer_idx=-1  # Ignored by compute_layerwise anyway
        )
        
        # MLP configuration (will be set when loading model)
        self.mlp_hidden_dim = None
        self.mlp_num_layers = None
        self.mlp_dropout = None
        self.mlp_input_mode = None
        
        # MLP encoder (the actual model we use)
        self.mlp_encoder = None
        self.model_path = model_path
        
        # Cache for embeddings
        self._embedding_dim = None
        
    def _extract_encoder_from_checkpoint(self, checkpoint: dict) -> MLPEncoder:
        """
        Extract and reconstruct the MLP encoder from a checkpoint.
        """
        # Get training hyperparameters
        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key. Cannot reconstruct model architecture.")
        
        saved_args = checkpoint['args']
        
        # Update configuration
        self.mlp_hidden_dim = saved_args['mlp_hidden_dim']
        self.mlp_num_layers = saved_args['mlp_layers']
        self.mlp_dropout = saved_args['dropout']
        self.mlp_input_mode = saved_args['mlp_input']
        
        # Get input dimension by running sample text through base model
        sample_texts = ["Sample text for dimension inference."]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        
        # Determine input dimension based on MLP input mode
        L, D = layerwise[0].shape
        if self.mlp_input_mode in ("last", "mean"):
            input_dim = D
        else:  # flatten
            input_dim = L * D
        
        # Create MLP encoder
        encoder = MLPEncoder(
            in_dim=input_dim,
            hidden_dim=self.mlp_hidden_dim,
            num_layers=self.mlp_num_layers,
            dropout=self.mlp_dropout,
        )
        
        # Extract encoder weights from classifier checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Filter out only encoder weights (remove 'linear.' prefix weights)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('enc.'):
                # Remove 'enc.' prefix to match MLPEncoder parameter names
                encoder_key = key[4:]  # Remove 'enc.'
                encoder_state_dict[encoder_key] = value
        
        # Load weights into encoder
        encoder.load_state_dict(encoder_state_dict)
        
        return encoder.to(self.device)
    
    def load_model(self, model_path: str):
        """Load a pre-trained MLP model and extract the encoder."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"Loading MLP encoder from {model_path}...")
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract encoder from classifier checkpoint
            self.mlp_encoder = self._extract_encoder_from_checkpoint(checkpoint)
            self.mlp_encoder.eval()
            
        except Exception as e:
            self.mlp_encoder = None
            raise RuntimeError(f"Failed to load MLP encoder from {model_path}: {e}")
        
        print(f"✓ Successfully loaded MLP encoder from {model_path}")
        print(f"  Architecture: {self.mlp_num_layers} layers, {self.mlp_hidden_dim} hidden_dim, {self.mlp_input_mode} input")
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
        Encode sentences using the MLP encoder directly.
        
        Args:
            sentences: Input sentences
            task_name: Name of the MTEB task (ignored)
            prompt_type: Prompt type for task (ignored)
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings [N, hidden_dim]
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
        if self.mlp_encoder is None:
            raise RuntimeError(
                "MLP encoder is not loaded! Call load_model() with a valid model path before encoding."
            )
        
        # Step 2: Create tensor dataset based on input mode
        dummy_labels = [0] * len(layerwise_embeddings)
        dataset = SimpleTensorDataset(
            layerwise_embeddings, 
            dummy_labels,
            mode=self.mlp_input_mode
        )
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Step 3: Run tensors through MLP encoder directly
        embeddings_list = []
        self.mlp_encoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch  # y is dummy labels, ignore
                X = X.to(self.device)
                # Direct call to MLP encoder - no classifier wrapper!
                mlp_embeddings = self.mlp_encoder(X)  # [B, hidden_dim]
                embeddings_list.append(mlp_embeddings.cpu().numpy())
        
        # Step 4: Concatenate and optionally normalize
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            if self.mlp_encoder is not None:
                self._embedding_dim = self.mlp_hidden_dim
            else:
                # Cannot determine without loading model
                raise RuntimeError("Cannot determine embedding dimension without loading a model first.")
        return self._embedding_dim
    
    def __repr__(self):
        return f"MLPWrapper(model={self.model_specs.model_family}-{self.model_specs.model_size}, mlp_dim={self.mlp_hidden_dim})"