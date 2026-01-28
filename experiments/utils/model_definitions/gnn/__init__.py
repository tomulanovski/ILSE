"""
GNN package for training on layer-wise embeddings.
"""

from .gnn_models import (
    build_edge_index,
    LayerGINEncoder,
    SingleClassifier,
    PairClassifier,
    PairCosineSimScore,
)

from .gnn_datasets import (
    SingleGraphDataset,
    PairGraphDataset,
    PairGraphRegressionDataset,
    pair_collate,
    pair_regression_collate,
    load_task_data,
    compute_layerwise,
)


__all__ = [
    # Models
    'build_edge_index',
    'LayerGINEncoder',
    'SingleClassifier',
    'PairClassifier',
    'PairCosineSimScore',
    
    # Datasets
    'SingleGraphDataset',
    'PairGraphDataset',
    'PairGraphRegressionDataset',
    'pair_collate',
    'pair_regression_collate',
    'load_task_data',
    'compute_layerwise',
    
    # Wrapper & Evaluation
    'CleanGNNWrapper',
    'CleanMLPWrapper',
]