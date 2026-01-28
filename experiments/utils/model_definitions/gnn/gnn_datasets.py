#!/usr/bin/env python3
"""
Dataset classes and data loading utilities for GNN training on layer-wise embeddings.
"""
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch_geometric.data import Data as GeomData
from torch_geometric.data import Batch as GeomBatch
import mteb

from ..text_automodel_wrapper import TextLayerwiseAutoModelWrapper
from .gnn_models import build_edge_index


class SimpleTensorDataset(torch.utils.data.Dataset):
    """
    Holds (X, y) where X is either:
      - last layer:          shape (D)
      - mean over layers:    shape (D)
      - flattened all layers:shape (L*D)
      - specific layer idx:  shape (D)
    """
    def __init__(self, layerwise_list, labels, mode: str = "last", layer_idx: int = None):
        """
        layerwise_list: list of np.ndarray each (L, D)
        mode: 'last' | 'mean' | 'flatten' | 'layer'
        layer_idx: required if mode='layer', specifies which layer to use
        """
        Xs = []
        for arr in layerwise_list:
            arr = np.asarray(arr, dtype=np.float32)  # Ensure proper numpy array with float32
            L, D = arr.shape
            if mode == "last":
                Xs.append(arr[-1])          # (D,)
            elif mode == "mean":
                Xs.append(arr.mean(axis=0)) # (D,)
            elif mode == "flatten":
                Xs.append(arr.reshape(L * D)) # (L*D,)
            elif mode == "layer":
                if layer_idx is None:
                    raise ValueError("layer_idx must be specified when mode='layer'")
                if layer_idx < 0 or layer_idx >= L:
                    raise ValueError(f"layer_idx {layer_idx} out of range [0, {L})")
                Xs.append(arr[layer_idx])   # (D,)
            else:
                raise ValueError(f"Unknown mlp_input mode: {mode}")
        # Use torch.tensor() instead of torch.from_numpy() to avoid numpy type issues
        self.X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in Xs])
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class LayerwiseTensorDataset(torch.utils.data.Dataset):
    """
    Dataset for learned weighting encoder.
    Keeps all layer embeddings: shape (L, D) for each example.
    """
    def __init__(self, layerwise_list, labels):
        """
        Args:
            layerwise_list: list of np.ndarray each of shape (L, D)
            labels: list of class labels
        """
        # Stack into [num_examples, num_layers, layer_dim]
        # Use torch.tensor() instead of torch.from_numpy() to avoid numpy type issues
        self.X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in layerwise_list])
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # X[idx] is [num_layers, layer_dim]


class PairTensorRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for STS regression tasks using MLP models on tensor representations.
    Each item is (X1, X2, score), where X1 and X2 are (L, D) arrays (e.g., transformer layers).
    """

    def __init__(self, layerwise_list_a, layerwise_list_b, scores, mode="last"):
        """
        Args:
            layerwise_list_a: list of np.ndarray of shape (L, D)
            layerwise_list_b: same as above
            scores: list of floats (regression targets)
            mode: how to combine the two layerwise tensors:
                  - 'last': concatenate last layer of both
                  - 'mean': concatenate mean-pooled layerwise
                  - 'diff': absolute difference of last layer
                  - 'mul': elementwise multiplication of last layer
                  - 'all': concat(last1, last2, |diff|, last1*last2)
        """
        assert len(layerwise_list_a) == len(layerwise_list_b) == len(scores)
        self.X1 = layerwise_list_a
        self.X2 = layerwise_list_b
        self.scores = scores
        self.mode = mode

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]

        if self.mode == "last":
            xa = x1[-1]
            xb = x2[-1]
        elif self.mode == "mean":
            xa = x1.mean(axis=0)
            xb = x2.mean(axis=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Use torch.tensor() to avoid numpy type issues
        return (
        torch.tensor(xa, dtype=torch.float32),
        torch.tensor(xb, dtype=torch.float32),
        torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        )


class PairLayerwiseRegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for STS regression with Weighted encoder.
    Keeps full layerwise embeddings [L,D] for both sentences.
    Returns (x1, x2, score) where x1, x2 are [L, D] tensors.
    """
    def __init__(self, layerwise_list_a, layerwise_list_b, scores):
        assert len(layerwise_list_a) == len(layerwise_list_b) == len(scores)
        self.X1 = layerwise_list_a  # List of [L, D] arrays
        self.X2 = layerwise_list_b
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Use torch.tensor() to avoid numpy type issues
        x1 = torch.tensor(self.X1[idx], dtype=torch.float32)  # [L, D]
        x2 = torch.tensor(self.X2[idx], dtype=torch.float32)  # [L, D]
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        return x1, x2, score


class SingleGraphDataset(Dataset):
    """Dataset for single text classification tasks. Each item is (Graph over layers for one text, label)."""

    def __init__(
        self,
        layerwise_list: List[np.ndarray],  # list of (L, D) arrays
        labels: List[int],
        graph_type: str = "fully_connected",
        cayley_jumps: Optional[List[int]] = None,
        add_virtual_node: bool = False,
        keep_embedding_layer: bool = False,  # Always keep embedding layer, never remove it
    ):
        self.items = layerwise_list
        self.labels = labels
        self.graph_type = graph_type
        self.cayley_jumps = cayley_jumps

        # Precompute shared edge_index shape for performance (L may vary? assume fixed per model)
        if len(self.items) == 0:
            raise ValueError("Cannot create dataset with empty layerwise_list")
        
        L = self.items[0].shape[0]
        # Validate that all items have the same number of layers
        for i, item in enumerate(self.items):
            if item.shape[0] != L:
                raise ValueError(f"Inconsistent number of layers: item 0 has {L} layers, item {i} has {item.shape[0]} layers")
        # Handle cayley: check if L = (cayley_size + 1), meaning we have embedding + transformer layers
        if self.graph_type == "cayley":
            from .gnn_models import check_if_cayley_size_plus_one, find_minimal_cayley_n
            is_match, n, cayley_size = check_if_cayley_size_plus_one(L)

            # Keep embedding layer and pad with virtual nodes
            if keep_embedding_layer:
                # Never remove embedding layer, always use adaptive approach
                self.items = [np.asarray(item) for item in self.items]
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley: {L} layers (keeping all including embedding), using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
            elif is_match:
                # L equals (cayley_size + 1), so remove first layer (embedding) and use n
                print(f"cayley (legacy): Detected {L} layers = {cayley_size} + 1 (embedding). Removing first layer, using SL(2, Z_{n}) with {cayley_size} nodes.")
                self.items = [np.asarray(item[1:]) for item in self.items]  # Slice [1:] to remove embedding layer
                L = cayley_size
                self.cayley_num_nodes = cayley_size  # Exact match, no virtual nodes
            else:
                # L doesn't match (cayley_size + 1), keep all layers and use adaptive approach
                self.items = [np.asarray(item) for item in self.items]
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley (legacy): {L} layers, using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
        else:
            # Ensure all items are proper numpy arrays
            self.items = [np.asarray(item) for item in self.items]
            self.cayley_num_nodes = None  # Not used for other graph types

        if self.graph_type == "virtual_node":
            self.edge_index = build_edge_index(L, "virtual_node", cayley_jumps=self.cayley_jumps)
        elif self.graph_type == "cayley":
            # Build edge index for full Cayley graph (including virtual nodes)
            self.edge_index = build_edge_index(self.cayley_num_nodes, "cayley", cayley_jumps=self.cayley_jumps)
        else:
            self.edge_index = build_edge_index(L, self.graph_type, cayley_jumps=self.cayley_jumps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Use torch.tensor() to avoid numpy type issues
        x = torch.tensor(self.items[idx], dtype=torch.float32)  # [L, D]

        if self.graph_type == "cayley":
            # Add virtual nodes if Cayley graph requires more nodes than we have layers
            num_real_layers = x.size(0)
            num_virtual = self.cayley_num_nodes - num_real_layers
            if num_virtual > 0:
                # Initialize virtual nodes as zeros
                virt = torch.zeros(num_virtual, x.size(1), dtype=x.dtype)
                x = torch.cat([x, virt], dim=0)  # [cayley_num_nodes, D]

        y = int(self.labels[idx])
        data = GeomData(x=x, edge_index=self.edge_index.clone())
        data.y = torch.tensor(y, dtype=torch.long)

        # For cayley: track number of real nodes (for optional pooling control)
        if self.graph_type == "cayley":
            num_real_layers = self.items[idx].shape[0]  # Original layer count
            data.num_real_nodes = num_real_layers
        else:
            data.num_real_nodes = x.size(0)  # All nodes are real

        return data


class PairGraphDataset(Dataset):
    """Dataset for pair classification tasks. Each item is (Graph(text1), Graph(text2), label)."""

    def __init__(
        self,
        layerwise_list_a: List[np.ndarray],  # list of (L, D)
        layerwise_list_b: List[np.ndarray],  # list of (L, D)
        labels: List[int],
        graph_type: str = "fully_connected",
        cayley_jumps: Optional[List[int]] = None,
        add_virtual_node: bool = False,
        keep_embedding_layer: bool = False,  # Always keep embedding layer, never remove it
    ):
        assert len(layerwise_list_a) == len(layerwise_list_b) == len(labels)
        if len(layerwise_list_a) == 0 or len(layerwise_list_b) == 0:
            raise ValueError("Cannot create dataset with empty layerwise lists")
            
        self.A = layerwise_list_a
        self.B = layerwise_list_b
        self.labels = labels
        self.graph_type = graph_type
        self.cayley_jumps = cayley_jumps
        self.add_virtual_node = add_virtual_node

        L = self.A[0].shape[0]
        # Validate layer consistency
        for i, (item_a, item_b) in enumerate(zip(self.A, self.B)):
            if item_a.shape[0] != L or item_b.shape[0] != L:
                raise ValueError(f"Inconsistent layer dimensions at index {i}: expected {L}, got {item_a.shape[0]}, {item_b.shape[0]}")
        # Handle cayley: check if L = (cayley_size + 1), meaning we have embedding + transformer layers
        if self.graph_type == "cayley":
            from .gnn_models import check_if_cayley_size_plus_one, find_minimal_cayley_n
            is_match, n, cayley_size = check_if_cayley_size_plus_one(L)

            # Keep embedding layer and pad with virtual nodes
            if keep_embedding_layer:
                # Never remove embedding layer, always use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley: {L} layers (keeping all including embedding), using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
            elif is_match:
                # L equals (cayley_size + 1), so remove first layer (embedding) and use n
                print(f"cayley (legacy): Detected {L} layers = {cayley_size} + 1 (embedding). Removing first layer, using SL(2, Z_{n}) with {cayley_size} nodes.")
                self.A = [np.asarray(item[1:]) for item in self.A]  # Slice [1:] to remove embedding layer
                self.B = [np.asarray(item[1:]) for item in self.B]
                L = cayley_size
                self.cayley_num_nodes = cayley_size  # Exact match, no virtual nodes
            else:
                # L doesn't match (cayley_size + 1), keep all layers and use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley (legacy): {L} layers, using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
        else:
            self.cayley_num_nodes = None  # Not used for other graph types

        if self.graph_type == "cayley":
            # Build edge index for full Cayley graph (including virtual nodes)
            self.edge_index = build_edge_index(self.cayley_num_nodes, "cayley", cayley_jumps=self.cayley_jumps, include_virtual=False)
        else:
            self.edge_index = build_edge_index(L, self.graph_type, cayley_jumps=self.cayley_jumps, include_virtual=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Use torch.tensor() to avoid numpy type issues
        xa = torch.tensor(self.A[idx], dtype=torch.float32)  # [L, D]
        xb = torch.tensor(self.B[idx], dtype=torch.float32)  # [L, D]

        if False:  # virtual_node removed
            zeros_a = torch.zeros(1, xa.size(1), dtype=xa.dtype)
            zeros_b = torch.zeros(1, xb.size(1), dtype=xb.dtype)
            xa = torch.cat([xa, zeros_a], dim=0)
            xb = torch.cat([xb, zeros_b], dim=0)

        elif self.graph_type == "cayley":
            # Add virtual nodes if Cayley graph requires more nodes than we have layers
            num_real_layers = xa.size(0)
            num_virtual = self.cayley_num_nodes - num_real_layers
            if num_virtual > 0:
                virt_a = torch.zeros(num_virtual, xa.size(1), dtype=xa.dtype)
                virt_b = torch.zeros(num_virtual, xb.size(1), dtype=xb.dtype)
                xa = torch.cat([xa, virt_a], dim=0)  # [cayley_num_nodes, D]
                xb = torch.cat([xb, virt_b], dim=0)  # [cayley_num_nodes, D]

        ya = GeomData(x=xa, edge_index=self.edge_index.clone())
        yb = GeomData(x=xb, edge_index=self.edge_index.clone())
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        # For cayley: track number of real nodes (for optional pooling control)
        if self.graph_type == "cayley":
            num_real_layers = self.A[idx].shape[0]  # Original layer count (before virtual nodes added)
            ya.num_real_nodes = num_real_layers
            yb.num_real_nodes = num_real_layers
        else:
            ya.num_real_nodes = xa.size(0)  # All nodes are real
            yb.num_real_nodes = xb.size(0)

        return ya, yb, y


class PairGraphRegressionDataset(Dataset):
    """Dataset for pair regression tasks (STS). Each item is (Graph(text1), Graph(text2), score)."""
    
    def __init__(
        self,
        layerwise_list_a: List[np.ndarray],  # list of (L, D)
        layerwise_list_b: List[np.ndarray],  # list of (L, D)
        scores: List[float],  # regression targets
        graph_type: str = "fully_connected",
        cayley_jumps: Optional[List[int]] = None,        
    ):
        assert len(layerwise_list_a) == len(layerwise_list_b) == len(scores)
        if len(layerwise_list_a) == 0 or len(layerwise_list_b) == 0:
            raise ValueError("Cannot create dataset with empty layerwise lists")
            
        self.A = layerwise_list_a
        self.B = layerwise_list_b
        self.scores = scores
        self.graph_type = graph_type
        self.cayley_jumps = cayley_jumps        

        L = self.A[0].shape[0]
        # Validate layer consistency
        for i, (item_a, item_b) in enumerate(zip(self.A, self.B)):
            if item_a.shape[0] != L or item_b.shape[0] != L:
                raise ValueError(f"Inconsistent layer dimensions at index {i}: expected {L}, got {item_a.shape[0]}, {item_b.shape[0]}")
        # Handle cayley: check if L = (cayley_size + 1), meaning we have embedding + transformer layers
        if self.graph_type == "cayley":
            from .gnn_models import check_if_cayley_size_plus_one, find_minimal_cayley_n
            is_match, n, cayley_size = check_if_cayley_size_plus_one(L)

            # Keep embedding layer and pad with virtual nodes
            if keep_embedding_layer:
                # Never remove embedding layer, always use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley: {L} layers (keeping all including embedding), using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
            elif is_match:
                # L equals (cayley_size + 1), so remove first layer (embedding) and use n
                print(f"cayley (legacy): Detected {L} layers = {cayley_size} + 1 (embedding). Removing first layer, using SL(2, Z_{n}) with {cayley_size} nodes.")
                self.A = [np.asarray(item[1:]) for item in self.A]  # Slice [1:] to remove embedding layer
                self.B = [np.asarray(item[1:]) for item in self.B]
                L = cayley_size
                self.cayley_num_nodes = cayley_size  # Exact match, no virtual nodes
            else:
                # L doesn't match (cayley_size + 1), keep all layers and use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(L)
                print(f"cayley (legacy): {L} layers, using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - L} virtual nodes).")
        else:
            self.cayley_num_nodes = None  # Not used for other graph types

        if self.graph_type == "virtual_node":
            self.edge_index = build_edge_index(L, "virtual_node", cayley_jumps=self.cayley_jumps)
        elif self.graph_type == "cayley":
            # Build edge index for full Cayley graph (including virtual nodes)
            self.edge_index = build_edge_index(self.cayley_num_nodes, "cayley", cayley_jumps=self.cayley_jumps)
        else:
            self.edge_index = build_edge_index(L, self.graph_type, cayley_jumps=self.cayley_jumps)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Use torch.tensor() to avoid numpy type issues
        xa = torch.tensor(self.A[idx], dtype=torch.float32)  # [L, D]
        xb = torch.tensor(self.B[idx], dtype=torch.float32)  # [L, D]

        if self.graph_type == "virtual_node":
            zeros_a = torch.zeros(1, xa.size(1), dtype=xa.dtype)
            zeros_b = torch.zeros(1, xb.size(1), dtype=xb.dtype)
            xa = torch.cat([xa, zeros_a], dim=0)
            xb = torch.cat([xb, zeros_b], dim=0)

        elif self.graph_type == "cayley":
            # Add virtual nodes if Cayley graph requires more nodes than we have layers
            num_real_layers = xa.size(0)
            num_virtual = self.cayley_num_nodes - num_real_layers
            if num_virtual > 0:
                virt_a = torch.zeros(num_virtual, xa.size(1), dtype=xa.dtype)
                virt_b = torch.zeros(num_virtual, xb.size(1), dtype=xb.dtype)
                xa = torch.cat([xa, virt_a], dim=0)  # [cayley_num_nodes, D]
                xb = torch.cat([xb, virt_b], dim=0)  # [cayley_num_nodes, D]

        ya = GeomData(x=xa, edge_index=self.edge_index.clone())
        yb = GeomData(x=xb, edge_index=self.edge_index.clone())
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)

        # For cayley: track number of real nodes (for optional pooling control)
        if self.graph_type == "cayley":
            num_real_layers = self.A[idx].shape[0]  # Original layer count (before virtual nodes added)
            ya.num_real_nodes = num_real_layers
            yb.num_real_nodes = num_real_layers
        else:
            ya.num_real_nodes = xa.size(0)  # All nodes are real
            yb.num_real_nodes = xb.size(0)

        return ya, yb, score


def single_collate(batch):
    """Custom collate function for single graph datasets."""
    return GeomBatch.from_data_list(batch)


def pair_collate(batch):
    """Custom collate function for pair datasets."""
    a_list, b_list, y_list = zip(*batch)
    return (
        GeomBatch.from_data_list(list(a_list)),
        GeomBatch.from_data_list(list(b_list)),
        torch.stack(list(y_list)),
    )


def pair_regression_collate(batch):
    """Custom collate function for pair regression datasets."""
    a_list, b_list, score_list = zip(*batch)
    return (
        GeomBatch.from_data_list(list(a_list)),
        GeomBatch.from_data_list(list(b_list)),
        torch.stack(list(score_list)),
    )


def load_task_data(task_name: str, split: str = "train") -> Dict[str, Any]:
    """
    Load data for training from MTEB tasks.
    
    Args:
        task_name: MTEB task name
        split: Data split to load ('train', 'validation', 'test')
        
    Returns:
        Dict describing the data for training with keys:
        - type: "single" or "pair"  
        - text/text_a/text_b: text data
        - labels: labels
        - num_classes: number of classes
        - scores: regression scores (for STS tasks)
    
    Supported tasks:
      Classification (single text):
      - AmazonCounterfactualClassification
      - Banking77Classification  
      - MTOPIntentClassification
      - EmotionClassification
      - MassiveIntentClassification
      - MTOPDomainClassification
      - MassiveScenarioClassification
      
      STS (pair regression):
      - SICK-R
      - STSBenchmark
      - BIOSSES
    """
    task_name = task_name.strip()

    try:
        task = mteb.get_task(task_name)
        
        # Get HuggingFace dataset info from metadata
        dataset_info = getattr(task.metadata, 'dataset', None)
        
        if not dataset_info:
            raise ValueError(f"No HuggingFace dataset info found for task '{task_name}'")
        
        # Extract dataset path and config
        if isinstance(dataset_info, dict):
            hf_path = dataset_info.get('path')
            hf_config = dataset_info.get('name')
            hf_revision = dataset_info.get('revision')
        elif isinstance(dataset_info, str):
            hf_path = dataset_info
            hf_config = None
            hf_revision = None
        else:
            raise ValueError(f"Invalid dataset info format for task '{task_name}'")
        
        print(f"Loading {task_name} from HuggingFace: {hf_path}")
        if hf_config:
            print(f"  Config: {hf_config}")
        
        # Load dataset from HuggingFace
        load_kwargs = {}
        if hf_config:
            load_kwargs['name'] = hf_config
        if hf_revision:
            load_kwargs['revision'] = hf_revision
            
        dataset = load_dataset(hf_path, **load_kwargs)
        
        # Check for English language key
        if 'en' in dataset:
            print(f"Found 'en' language key")
            dataset = dataset['en']
        elif 'eng' in dataset:
            print(f"Found 'eng' language key")
            dataset = dataset['eng']

        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{split}' not available for task '{task_name}'. Available splits: {available_splits}")
        
        split_data = dataset[split]
        print(f"Loaded {len(split_data)} examples from {split} split")

    except ImportError as e:
        raise ImportError(f"Required libraries not installed: {e}. Install with: pip install mteb datasets")
    except Exception as e:
        raise RuntimeError(f"Error loading task '{task_name}': {e}")
    
    columns = split_data.column_names
    
    if task.metadata.type.lower() == "classification":
        if 'text' in columns:
            texts = split_data['text']
        elif 'verse_text' in columns:
            texts = split_data['verse_text']
        else:
            raise ValueError(f"Dataset has unknown 'text' like column. Found: {columns}")
        
        labels = split_data['label']

        # Convert string labels to integers if necessary
        if labels and isinstance(labels[0], str):
            # Create label to integer mapping
            unique_labels = sorted(set(labels))
            label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_int[label] for label in labels]
            num_classes = len(unique_labels)
        else:
            # Labels are already integers
            unique_labels = set(labels)
            num_classes = len(unique_labels)

        return {
            "type": "single",
            "text": texts,
            "labels": labels,
            "num_classes": num_classes,
            "task_type": "classification"
        }
                    
    elif task.metadata.type.lower() == "sts":
        # Semantic textual similarity (pair regression) - MTEB methodology
        sentence1 = split_data['sentence1']
        sentence2 = split_data['sentence2'] 
        scores = split_data['score']
        
        # Normalize scores to [0, 1] range for training stability
        if task_name == "SICK-R":
            # SICK-R scores are 1-5
            min_score, max_score = 1.0, 5.0
        elif task_name == "STSBenchmark":
            # STS-B scores are 0-5
            min_score, max_score = 0.0, 5.0
        elif task_name == "BIOSSES":
            # BIOSSES scores are 0-4
            min_score, max_score = 0.0, 4.0
        else:
            # Auto-detect range
            min_score, max_score = float(min(scores)), float(max(scores))
        
        # Normalize scores to [0, 1]
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        return {
            "type": "pair_regression",
            "text_a": sentence1,
            "text_b": sentence2, 
            "scores": normalized_scores,  # Normalized for training
            "original_scores": scores,    # Original for MTEB evaluation
            "min_score": min_score,
            "max_score": max_score,
            "task_type": "sts_regression"
        }
        
    else:
        raise ValueError(f"Unsupported task '{task_name}'. Task type: {task.metadata.type}")


def compute_layerwise(
    wrapper: TextLayerwiseAutoModelWrapper,
    texts: List[str],
    batch_size: Optional[int] = None,
    token_pooling_method: str = "mean",
) -> List[np.ndarray]:
    """
    Compute layerwise embeddings for texts using the pre-trained model wrapper.
    
    Args:
        wrapper: Pre-trained model wrapper
        texts: List of texts to encode
        batch_size: Batch size for encoding
        token_pooling_method: Method for pooling tokens within each layer
        
    Returns:
        List of (L, D) arrays: per sample, the pooled hidden state per layer.
    """
    # Ensure we get layerwise encodings
    result = wrapper.encode(
        texts,
        return_raw_hidden_states=True,
        pooling_method=token_pooling_method,
        batch_size=batch_size,
        verbose=True,
    )
    _, _, layerwise_enc = result  # (L, N, D)
    if isinstance(layerwise_enc, torch.Tensor):
        layerwise_enc = layerwise_enc.cpu().numpy()

    # Clear GPU cache immediately after extraction to prevent accumulation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # transpose to list of (L, D) per sample
    if len(layerwise_enc.shape) == 2:
        # Single sample case: shape is (L, D)
        out = [layerwise_enc.copy()]
    else:
        # Multiple samples case: shape is (L, N, D)
        L, N, D = layerwise_enc.shape
        out = [layerwise_enc[:, i, :].copy() for i in range(N)]
    return out


# ============================================================================
# Precomputed STS Dataset Classes (for faster training)
# ============================================================================

class PrecomputedSTSGraphDataset(Dataset):
    """
    Dataset for STS regression with precomputed embeddings (GIN models).
    Loads sentence pair embeddings from HDF5 and creates graph representations.
    """
    def __init__(self, h5_path: str, split: str = "train", graph_type: str = "fully_connected", cayley_jumps: Optional[List[int]] = None, keep_embedding_layer: bool = False):
        """
        Args:
            h5_path: Path to HDF5 file with precomputed STS embeddings
            split: 'train', 'validation', or 'test'
            graph_type: Graph structure ('fully_connected', 'cayley')
            cayley_jumps: Jump distances for Cayley graph
            keep_embedding_layer: Always keep embedding layer, never remove it
        """
        import h5py

        self.h5_path = h5_path
        self.split = split
        self.graph_type = graph_type
        self.cayley_jumps = cayley_jumps or [1, 3]

        # Load embeddings from HDF5
        with h5py.File(h5_path, 'r') as f:
            self.embeddings_a = f[f'{split}/embeddings_a'][:]  # [N, L, D]
            self.embeddings_b = f[f'{split}/embeddings_b'][:]  # [N, L, D]
            self.scores = f[f'{split}/scores'][:]  # [N]

            # Get dimensions
            self.num_samples, self.num_layers, self.layer_dim = self.embeddings_a.shape

        # Handle cayley: check if num_layers = (cayley_size + 1), meaning we have embedding + transformer layers
        if self.graph_type == "cayley":
            from .gnn_models import check_if_cayley_size_plus_one, find_minimal_cayley_n
            is_match, n, cayley_size = check_if_cayley_size_plus_one(self.num_layers)

            # Keep embedding layer and pad with virtual nodes
            if keep_embedding_layer:
                # Never remove embedding layer, always use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(self.num_layers)
                print(f"cayley: {self.num_layers} layers (keeping all including embedding), using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - self.num_layers} virtual nodes).")
            elif is_match:
                # num_layers equals (cayley_size + 1), so remove first layer (embedding) and use n
                print(f"cayley (legacy): Detected {self.num_layers} layers = {cayley_size} + 1 (embedding). Removing first layer, using SL(2, Z_{n}) with {cayley_size} nodes.")
                self.embeddings_a = self.embeddings_a[:, 1:, :]  # Slice [:, 1:, :] to remove embedding layer
                self.embeddings_b = self.embeddings_b[:, 1:, :]
                self.num_layers = cayley_size
                self.cayley_num_nodes = cayley_size
            else:
                # num_layers doesn't match (cayley_size + 1), keep all layers and use adaptive approach
                _, self.cayley_num_nodes = find_minimal_cayley_n(self.num_layers)
                print(f"cayley (legacy): {self.num_layers} layers, using adaptive SL(2, Z_n) with {self.cayley_num_nodes} nodes ({self.cayley_num_nodes - self.num_layers} virtual nodes).")

        # Build edge index for graph construction
        if self.graph_type == "cayley":
            self.edge_index = build_edge_index(self.cayley_num_nodes, "cayley", cayley_jumps=self.cayley_jumps)
        else:
            self.edge_index = build_edge_index(self.num_layers, self.graph_type, cayley_jumps=self.cayley_jumps)

        print(f"Loaded {self.num_samples} {split} pairs from {h5_path}")
        print(f"  Shape: {self.num_layers} layers × {self.layer_dim} dims")
        print(f"  Graph type: {self.graph_type}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xa = torch.tensor(self.embeddings_a[idx], dtype=torch.float32)  # [L, D]
        xb = torch.tensor(self.embeddings_b[idx], dtype=torch.float32)  # [L, D]

        # Add virtual node if needed
        # Add virtual nodes for cayley if needed
        if self.graph_type == "cayley":
            num_real_layers = xa.size(0)
            num_virtual = self.cayley_num_nodes - num_real_layers
            if num_virtual > 0:
                virt_a = torch.zeros(num_virtual, xa.size(1), dtype=xa.dtype)
                virt_b = torch.zeros(num_virtual, xb.size(1), dtype=xb.dtype)
                xa = torch.cat([xa, virt_a], dim=0)  # [cayley_num_nodes, D]
                xb = torch.cat([xb, virt_b], dim=0)  # [cayley_num_nodes, D]

        # Create PyG Data objects
        ya = GeomData(x=xa, edge_index=self.edge_index.clone())
        yb = GeomData(x=xb, edge_index=self.edge_index.clone())
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)

        # For cayley: track number of real nodes (for optional pooling control)
        if self.graph_type == "cayley":
            # embeddings_a/b were potentially sliced to remove embedding layer during __init__
            # After slicing, self.num_layers reflects the actual layer count
            # num_real_nodes is self.num_layers (before virtual nodes are added in __getitem__)
            num_real_layers = self.num_layers if not hasattr(self, 'cayley_num_nodes') else self.num_layers
            # Actually, self.embeddings_a[idx] has already been sliced, so its shape[0] is the real layer count
            num_real_layers_actual = self.embeddings_a[idx].shape[0]
            ya.num_real_nodes = num_real_layers_actual
            yb.num_real_nodes = num_real_layers_actual
        else:
            ya.num_real_nodes = xa.size(0)  # All nodes are real
            yb.num_real_nodes = xb.size(0)

        return ya, yb, score


class PrecomputedSTSTensorDataset(Dataset):
    """
    Dataset for STS regression with precomputed embeddings (MLP models).
    Loads sentence pair embeddings from HDF5 and creates tensor representations.
    """
    def __init__(self, h5_path: str, split: str = "train", mode: str = "last"):
        """
        Args:
            h5_path: Path to HDF5 file with precomputed STS embeddings
            split: 'train', 'validation', or 'test'
            mode: 'last' (last layer), 'mean' (average layers), 'flatten' (concatenate all layers)
        """
        import h5py

        self.h5_path = h5_path
        self.split = split
        self.mode = mode

        # Load embeddings from HDF5
        with h5py.File(h5_path, 'r') as f:
            embeddings_a = f[f'{split}/embeddings_a'][:]  # [N, L, D]
            embeddings_b = f[f'{split}/embeddings_b'][:]  # [N, L, D]
            self.scores = f[f'{split}/scores'][:]  # [N]

            N, L, D = embeddings_a.shape

        # Process embeddings according to mode
        if mode == "last":
            self.X_a = embeddings_a[:, -1, :]  # [N, D]
            self.X_b = embeddings_b[:, -1, :]  # [N, D]
        elif mode == "mean":
            self.X_a = embeddings_a.mean(axis=1)  # [N, D]
            self.X_b = embeddings_b.mean(axis=1)  # [N, D]
        elif mode == "flatten":
            self.X_a = embeddings_a.reshape(N, L * D)  # [N, L*D]
            self.X_b = embeddings_b.reshape(N, L * D)  # [N, L*D]
        else:
            raise ValueError(f"Unknown mode: {mode}. Choices: 'last', 'mean', 'flatten'")

        self.num_samples = N
        self.input_dim = self.X_a.shape[1]  # Dimension after mode processing
        print(f"Loaded {N} {split} pairs from {h5_path}")
        print(f"  Mode: {mode}, Shape: {self.X_a.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xa = torch.tensor(self.X_a[idx], dtype=torch.float32)
        xb = torch.tensor(self.X_b[idx], dtype=torch.float32)
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        return xa, xb, score


class PrecomputedSTSLayerwiseDataset(Dataset):
    """
    Dataset for STS regression with precomputed embeddings (Weighted models).
    Loads sentence pair embeddings from HDF5 and keeps full layerwise information.
    """
    def __init__(self, h5_path: str, split: str = "train"):
        """
        Args:
            h5_path: Path to HDF5 file with precomputed STS embeddings
            split: 'train', 'validation', or 'test'
        """
        import h5py

        self.h5_path = h5_path
        self.split = split

        # Load embeddings from HDF5
        with h5py.File(h5_path, 'r') as f:
            self.embeddings_a = f[f'{split}/embeddings_a'][:]  # [N, L, D]
            self.embeddings_b = f[f'{split}/embeddings_b'][:]  # [N, L, D]
            self.scores = f[f'{split}/scores'][:]  # [N]

            self.num_samples, self.num_layers, self.layer_dim = self.embeddings_a.shape

        print(f"Loaded {self.num_samples} {split} pairs from {h5_path}")
        print(f"  Shape: {self.num_layers} layers × {self.layer_dim} dims")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return full layerwise embeddings [L, D] for both sentences
        xa = torch.tensor(self.embeddings_a[idx], dtype=torch.float32)  # [L, D]
        xb = torch.tensor(self.embeddings_b[idx], dtype=torch.float32)  # [L, D]
        score = torch.tensor(float(self.scores[idx]), dtype=torch.float32)
        return xa, xb, score