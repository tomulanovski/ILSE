#!/usr/bin/env python3
"""
 MTEB-compatible wrapper for trained GNN models.
Directly uses LayerGINEncoder without unnecessary classifier wrappers.
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
from experiments.utils.model_definitions.gnn.gnn_models import LayerGINEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import compute_layerwise
from experiments.utils.model_definitions.gnn.gnn_datasets import SingleGraphDataset


class GNNWrapper:
    """
     MTEB-compatible wrapper for GNN models trained on layer-wise embeddings.
    
    Directly uses LayerGINEncoder without classifier wrapper overhead.
    """
    
    def __init__(
        self,
        model_specs: TextModelSpecifications,
        device_map: str = "auto",
        model_path: Optional[str] = None,
    ):
        """
        Initialize GNN wrapper.
        
        Args:
            model_specs: Specifications for the base pre-trained model
            device_map: Device mapping for the base model
            model_path: Path to pre-trained GNN model (if available)
        """
        self.model_specs = model_specs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the base pre-trained model wrapper
        self.base_wrapper = TextLayerwiseAutoModelWrapper(
            model_specs, 
            device_map=device_map,
            evaluation_layer_idx=-1  # Ignored by compute_layerwise anyway
        )
        
        # GNN configuration (will be set when loading model)
        self.gnn_hidden_dim = None
        self.gnn_num_layers = None
        self.gnn_dropout = None
        self.graph_type = None
        self.cayley_jumps = None
        self.use_virtual_node = None
        
        # GNN encoder (the actual model we use)
        self.gnn_encoder = None
        self.model_path = model_path
        
        # Cache for embeddings
        self._embedding_dim = None
        
    def _extract_encoder_from_checkpoint(self, checkpoint: dict) -> LayerGINEncoder:
        """
        Extract and reconstruct the GNN encoder from a checkpoint.
        Handles both SingleClassifier and PairClassifier checkpoints.
        """
        # Get training hyperparameters
        if 'args' not in checkpoint:
            raise ValueError("Checkpoint missing 'args' key. Cannot reconstruct model architecture.")
        
        saved_args = checkpoint['args']
        
        # Update configuration
        self.gnn_hidden_dim = saved_args['gin_hidden_dim']
        self.gnn_num_layers = saved_args['gin_layers']
        self.gnn_dropout = saved_args['dropout']
        self.graph_type = saved_args['graph_type']
        self.cayley_jumps = saved_args.get('cayley_jumps', [1, 3])
        self.use_virtual_node = (saved_args['graph_type'] == 'virtual_node')
        
        # Get input dimension by running sample text through base model
        sample_texts = ["Sample text for dimension inference."]
        layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
        input_dim = layerwise[0].shape[1]  # D dimension
        
        # Create GNN encoder
        encoder = LayerGINEncoder(
            in_dim=input_dim,
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
            dropout=self.gnn_dropout,
            gin_mlp_layers=saved_args.get('gin_mlp_layers', 1),
            node_to_choose=saved_args.get('node_to_choose', 'last'),
            graph_type=self.graph_type
        )
        
        # Extract encoder weights from classifier checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Filter out only encoder weights (remove 'linear.' prefix weights)
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('enc.'):
                # Remove 'enc.' prefix to match LayerGINEncoder parameter names
                encoder_key = key[4:]  # Remove 'enc.'
                
                # Handle legacy parameter naming
                if 'gin_layers' in encoder_key:
                    encoder_key = encoder_key.replace('gin_layers', 'gnn_layers')
                    
                encoder_state_dict[encoder_key] = value
        
        # Load weights into encoder
        encoder.load_state_dict(encoder_state_dict)
        
        return encoder.to(self.device)
    
    def load_model(self, model_path: str):
        """Load a pre-trained GNN model and extract the encoder."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        print(f"Loading GNN encoder from {model_path}...")
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract encoder from classifier checkpoint
            self.gnn_encoder = self._extract_encoder_from_checkpoint(checkpoint)
            self.gnn_encoder.eval()
            
        except Exception as e:
            self.gnn_encoder = None
            raise RuntimeError(f"Failed to load GNN encoder from {model_path}: {e}")
        
        print(f"✓ Successfully loaded GNN encoder from {model_path}")
        print(f"  Architecture: {self.gnn_num_layers} layers, {self.gnn_hidden_dim} hidden_dim, {self.graph_type} graph")
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
        Encode sentences using the GNN encoder directly.
        
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
        if self.gnn_encoder is None:
            raise RuntimeError(
                "GNN encoder is not loaded! Call load_model() with a valid model path before encoding."
            )
        
        # Step 2: Convert layerwise embeddings to graphs
        from torch_geometric.loader import DataLoader
        
        dummy_labels = [0] * len(layerwise_embeddings)
        dataset = SingleGraphDataset(
            layerwise_embeddings, 
            dummy_labels,
            graph_type=self.graph_type,
            cayley_jumps=self.cayley_jumps,
            add_virtual_node=self.use_virtual_node
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Step 3: Run graphs through GNN encoder directly
        embeddings_list = []
        self.gnn_encoder.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                # Direct call to GNN encoder - no classifier wrapper!
                graph_embeddings = self.gnn_encoder(batch)  # [B, hidden_dim]
                embeddings_list.append(graph_embeddings.cpu().numpy())
        
        # Step 4: Concatenate and optionally normalize
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)
            
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            if self.gnn_encoder is not None:
                self._embedding_dim = self.gnn_hidden_dim
            else:
                # Fallback: try to infer from base model
                sample_texts = ["Sample text for dimension inference."]
                layerwise = compute_layerwise(self.base_wrapper, sample_texts, batch_size=1)
                # This is the input dim, not output dim - we can't know output without loading model
                raise RuntimeError("Cannot determine embedding dimension without loading a model first.")
        return self._embedding_dim
    
    def __repr__(self):
        return f"GNNWrapper(model={self.model_specs.model_family}-{self.model_specs.model_size}, gnn_dim={self.gnn_hidden_dim})"