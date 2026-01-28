#!/usr/bin/env python3
"""
Core GNN models and graph construction utilities for layer-wise embeddings.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch as GeomBatch
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
import numpy as np
from collections import deque
# from torch_scatter import scatter_max


def get_cayley_graph(n):
    """
    Get the edge index of the Cayley graph (Cay(SL(2, Z_n); S_n)).
    
    When n=3, produces exactly 24 nodes (|SL(2, Z_3)| = 24).
    
    Args:
        n: Integer for Z_n (use n=3 for 24 nodes)
    
    Returns:
        edge_index: numpy array of shape [2, E] with edge connections
    """
    # Define 4 generators as 2x2 matrices
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n - 1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n - 1, 1]]
    ], dtype=np.int32)
    
    # Helper function to multiply 2x2 matrices mod n
    def mat_mult_mod(a, b, mod):
        """Multiply two 2x2 matrices modulo mod."""
        result = np.zeros((2, 2), dtype=np.int32)
        for i in range(2):
            for j in range(2):
                result[i, j] = (a[i, 0] * b[0, j] + a[i, 1] * b[1, j]) % mod
        return result
    
    # Helper function to convert matrix to tuple for hashing
    def mat_to_tuple(mat):
        return tuple(mat.flatten())
    
    # BFS from identity matrix to enumerate all group elements
    identity = np.array([[1, 0], [0, 1]], dtype=np.int32)
    queue = deque([identity])
    nodes = {mat_to_tuple(identity): 0}  # Map matrix → node index
    node_list = [identity]  # List of matrices in order
    
    while queue and len(nodes) < 1000:  # Safety limit
        current = queue.popleft()
        for gen in generators:
            new_matrix = mat_mult_mod(current, gen, n)
            new_tuple = mat_to_tuple(new_matrix)
            
            if new_tuple not in nodes:
                nodes[new_tuple] = len(nodes)
                node_list.append(new_matrix)
                queue.append(new_matrix)
    
    num_nodes = len(nodes)
    
    # Build edge index
    src_list, dst_list = [], []
    for node_idx in range(num_nodes):
        current_matrix = node_list[node_idx]
        for gen in generators:
            neighbor_matrix = mat_mult_mod(current_matrix, gen, n)
            neighbor_tuple = mat_to_tuple(neighbor_matrix)
            neighbor_idx = nodes[neighbor_tuple]
            
            # Add edge (undirected graph, add both directions)
            src_list.append(node_idx)
            dst_list.append(neighbor_idx)
            # Add reverse edge for undirected graph
            src_list.append(neighbor_idx)
            dst_list.append(node_idx)
    
    # Convert to numpy array format [2, E]
    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    return edge_index


def find_minimal_cayley_n(target_nodes: int, max_n: int = 10) -> tuple:
    """
    Find minimal n such that |SL(2, Z_n)| >= target_nodes.

    This allows cayley to adapt to any model size by finding a Cayley graph
    that's large enough, with extra nodes becoming virtual nodes.

    Args:
        target_nodes: Minimum number of nodes needed (e.g., number of layers)
        max_n: Maximum n to try (safety limit)

    Returns:
        (n, actual_nodes): Minimal n and the actual number of nodes in SL(2, Z_n)

    Examples:
        - TinyLlama (23 layers): returns (3, 24) - 1 virtual node
        - Pythia-410m (24 layers): returns (3, 24) - perfect fit
        - Pythia-2.8b (32 layers): returns (4, 48) - 16 virtual nodes
        - Llama3-8B (32 layers): returns (4, 48) - 16 virtual nodes
    """
    for n in range(2, max_n + 1):
        edge_index = get_cayley_graph(n)
        # Number of nodes is the max node index + 1
        actual_nodes = edge_index.max() + 1
        if actual_nodes >= target_nodes:
            return n, int(actual_nodes)

    raise ValueError(
        f"Could not find Cayley graph with >= {target_nodes} nodes within max_n={max_n}. "
        f"Try increasing max_n or use a different graph topology."
    )


def check_if_cayley_size_plus_one(num_nodes: int, max_n: int = 10) -> tuple:
    """
    Check if num_nodes equals (|SL(2, Z_n)| + 1) for some n.

    This is used for cayley to detect when we have embeddings + transformer layers.
    If true, we should remove the first layer (embedding) and use n.

    Args:
        num_nodes: Number of nodes to check (e.g., 25 for Pythia-410m with embedding)
        max_n: Maximum n to try

    Returns:
        (is_match, n, cayley_size):
            - is_match: True if num_nodes == cayley_size + 1
            - n: The n value that gives the match (or None)
            - cayley_size: The Cayley graph size (or None)

    Examples:
        - check_if_cayley_size_plus_one(25) → (True, 3, 24)  # Pythia-410m: 25 = 24 + 1
        - check_if_cayley_size_plus_one(24) → (False, None, None)  # 24 is exact match, not +1
    """
    for n in range(2, max_n + 1):
        edge_index = get_cayley_graph(n)
        cayley_size = edge_index.max() + 1
        if num_nodes == cayley_size + 1:
            return True, n, int(cayley_size)
    return False, None, None


def get_cayley_graph_sl2_24():
    """
    Wrapper for Pythia-410m: returns fixed 24-node Cayley graph.

    Uses SL(2, Z_3) which has exactly 24 elements, matching Pythia-410m's 24 layers.

    Returns:
        edge_index: torch.LongTensor of shape [2, E] with edge connections
    """
    edge_index_numpy = get_cayley_graph(3)  # n=3 gives 24 nodes
    # Convert to torch tensor format - use torch.tensor() for better compatibility
    return torch.tensor(edge_index_numpy, dtype=torch.long)


def build_hierarchical_edge_index(num_nodes: int, num_groups: int = 3) -> torch.LongTensor:
    """
    Build hierarchical graph topology for layer aggregation.

    Divides layers into groups (e.g., early/middle/late) with:
    - Dense connections within each group (all-to-all)
    - Sparse connections between adjacent groups (bridge connections)

    This topology captures the intuition that nearby layers are more related,
    while still allowing information flow across layer groups.

    Args:
        num_nodes: Number of nodes (layers), e.g., 24 for Pythia-410m
        num_groups: Number of hierarchical groups (default: 3 for early/middle/late)

    Returns:
        edge_index: torch.LongTensor of shape [2, E]

    Example for 24 layers with 3 groups:
        Group 0 (early):  layers 0-7   (dense internal connections)
        Group 1 (middle): layers 8-15  (dense internal connections)
        Group 2 (late):   layers 16-23 (dense internal connections)
        Bridge edges: 7↔8, 15↔16 (connecting adjacent groups)
    """
    device = torch.device("cpu")
    src_list, dst_list = [], []

    # Calculate group boundaries
    base_group_size = num_nodes // num_groups
    remainder = num_nodes % num_groups

    group_boundaries = []
    start = 0
    for g in range(num_groups):
        # Distribute remainder across first groups
        size = base_group_size + (1 if g < remainder else 0)
        end = start + size
        group_boundaries.append((start, end))
        start = end

    # Dense connections within each group (fully connected within group)
    for group_start, group_end in group_boundaries:
        for i in range(group_start, group_end):
            for j in range(group_start, group_end):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

    # Bridge connections between adjacent groups
    # Connect last node of group g to first node of group g+1 (bidirectional)
    for g in range(num_groups - 1):
        group_end_node = group_boundaries[g][1] - 1  # Last node of current group
        next_group_start_node = group_boundaries[g + 1][0]  # First node of next group

        # Bidirectional bridge
        src_list.extend([group_end_node, next_group_start_node])
        dst_list.extend([next_group_start_node, group_end_node])

        # Optional: also connect first of current to last of next for richer flow
        group_start_node = group_boundaries[g][0]
        next_group_end_node = group_boundaries[g + 1][1] - 1
        src_list.extend([group_start_node, next_group_end_node])
        dst_list.extend([next_group_end_node, group_start_node])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    return edge_index


def build_edge_index(
    num_nodes: int,
    graph_type: str = "fully_connected",
    cayley_jumps: Optional[List[int]] = None,
    num_hierarchical_groups: int = 3
) -> torch.LongTensor:
    """
    Build edge_index [2, E] for a graph whose nodes are LLM layers (0..L-1).

    Args:
        num_nodes: Number of nodes (layers)
        graph_type: 'fully_connected' | 'cayley' | 'hierarchical'
        cayley_jumps: used when graph_type == 'cayley'
        num_hierarchical_groups: Number of groups for hierarchical topology (default: 3)

    Returns:
        edge_index tensor of shape [2, E]
    """
    device = torch.device("cpu")

    if graph_type == "fully_connected":
        comb = torch.combinations(torch.arange(num_nodes, device=device), r=2).t()
        edge = torch.cat([comb, comb.flip(0)], dim=1)

    elif graph_type == "cayley":
        # Adaptive cayley: Find minimal Cayley graph size >= num_nodes
        # Extra nodes beyond num_nodes become virtual nodes that aid message passing
        n, cayley_nodes = find_minimal_cayley_n(num_nodes)
        edge_index_numpy = get_cayley_graph(n)
        edge = torch.tensor(edge_index_numpy, dtype=torch.long)

        # Log the virtual node count for transparency
        num_virtual = cayley_nodes - num_nodes
        if num_virtual > 0:
            print(f"cayley: Using SL(2, Z_{n}) with {cayley_nodes} nodes for {num_nodes} layers ({num_virtual} virtual nodes)")

    elif graph_type == "hierarchical":
        # Hierarchical topology: dense within groups, sparse between groups
        edge = build_hierarchical_edge_index(num_nodes, num_groups=num_hierarchical_groups)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return edge.long()


class LayerGINEncoder(nn.Module):
    """
    GIN encoder that takes a batch of layer-graphs (one per text), does message passing,
    and returns a per-graph embedding via global pooling (mean/sum/last).

    For Linear GCN Variant 1: set output_dim=num_classes (no separate classifier head needed)
    For Linear GCN Variant 2: set output_dim=hidden_dim (requires separate classifier head)

    Pooling options (node_to_choose):
    - 'mean': global mean pooling (default, most common)
    - 'sum': global sum pooling (preserves magnitude)
    - 'last': select last node after message passing

    Linear mode (use_linear=True):
    - Removes all ReLU activations for fair comparison with linear probing baselines
    - Applies to both GCN mode (gin_mlp_layers=0) and GIN mode (gin_mlp_layers>=1)
    - In GIN mode, removes ReLU from internal MLPs inside GINConv

    Cayley virtual nodes (pool_real_nodes_only=True):
    - Excludes virtual nodes from pooling, only pools over real layer nodes
    - Virtual nodes still participate in message passing

    Learnable epsilon (train_eps=True):
    - Learns self-loop weight in GIN aggregation: (1 + eps) * h_v + sum(neighbors)
    - Adds 1 learnable parameter per GIN layer (minimal overhead)
    - Can help when different layers need different self-attention weights
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        gin_mlp_layers: int = 1,
        node_to_choose: str = "mean",
        graph_type: str = None,  # REQUIRED - must be explicitly passed
        use_linear: bool = False,  # Remove all ReLU activations (applies to both GCN and GIN)
        output_dim: int = None,  # For direct num_classes output
        pool_real_nodes_only: bool = False,  # For cayley - exclude virtual nodes from pooling
        train_eps: bool = False  # Learnable epsilon in GIN aggregation (1 param per layer)
    ):
        super().__init__()
        if graph_type is None:
            raise ValueError("graph_type is required! Must be one of: fully_connected, cayley")
        self.node_to_choose = node_to_choose.lower()
        self.use_gcn = gin_mlp_layers == 0
        self.use_linear = use_linear  # Applies to both GCN and GIN modes
        self.graph_type = graph_type  # Store graph_type for evaluation
        self.pool_real_nodes_only = pool_real_nodes_only  # For cayley virtual node exclusion
        self.train_eps = train_eps  # Learnable epsilon in GIN (1 param per layer)

        # If output_dim is specified, use it; otherwise default to hidden_dim (backward compatible)
        self.output_dim = output_dim if output_dim is not None else hidden_dim

        self.proj_in = nn.Linear(in_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

        # Build GNN layers with small MLPs
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.use_gcn:
                self.gnn_layers.append(GCNConv(self.output_dim, self.output_dim))
            else:
                # GIN mode: build MLP inside GINConv
                mlp_modules = []
                for mlp_layer in range(gin_mlp_layers):
                    mlp_modules.append(nn.Linear(self.output_dim, self.output_dim))
                    if not self.use_linear:  # Skip ReLU in linear mode
                        mlp_modules.append(nn.ReLU())
                mlp = nn.Sequential(*mlp_modules)
                # train_eps: learnable epsilon for self-loop weight (adds 1 param per GIN layer)
                self.gnn_layers.append(GINConv(mlp, train_eps=self.train_eps))

        self.norms = nn.ModuleList([nn.LayerNorm(self.output_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()

    def forward(self, batch: GeomBatch) -> torch.Tensor:
        """
        Args:
            batch.x: [sum_nodes, F] - node features
            batch.edge_index: [2, sum_edges] - edge connections
            batch.batch: graph assignment vector

        Returns:
            [num_graphs, output_dim] - graph-level embeddings (output_dim = hidden_dim or num_classes)
        """
        x = self.proj_in(batch.x)
        # Skip ReLU in linear mode (applies to both GCN and GIN)
        if not self.use_linear:
            x = self.act(x)
        x = self.dropout(x)

        for conv, norm in zip(self.gnn_layers, self.norms):
            x = conv(x, batch.edge_index)
            x = norm(x)
            # Only apply activation for GCN mode when not in linear mode
            # (GIN mode has activation inside GINConv MLP, which is also controlled by use_linear)
            if self.use_gcn and not self.use_linear:
                x = self.act(x)
            x = self.dropout(x)

        # Pooling: optionally exclude virtual nodes for cayley
        if self.pool_real_nodes_only and hasattr(batch, 'num_real_nodes'):
            # Create mask for real nodes only (exclude virtual nodes)
            num_graphs = batch.num_graphs
            real_node_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

            for graph_idx in range(num_graphs):
                nodes_in_g = (batch.batch == graph_idx).nonzero(as_tuple=True)[0]
                num_real = batch.num_real_nodes[graph_idx].item()
                # First num_real nodes are real nodes, rest are virtual
                real_node_mask[nodes_in_g[:num_real]] = True

            # Filter to real nodes only
            x_pooling = x[real_node_mask]
            batch_pooling = batch.batch[real_node_mask]
        else:
            # Use all nodes for pooling
            x_pooling = x
            batch_pooling = batch.batch
            num_graphs = batch.num_graphs

        if self.node_to_choose == "mean":
            g = global_mean_pool(x_pooling, batch_pooling)

        elif self.node_to_choose == "sum":
            g = global_add_pool(x_pooling, batch_pooling)

        elif self.node_to_choose == "last":
            last_node_indices = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            for graph_idx in range(num_graphs):
                nodes_in_g = (batch_pooling == graph_idx).nonzero(as_tuple=True)[0]
                last_node_indices[graph_idx] = nodes_in_g[-1]   # take the last one
            g = x_pooling[last_node_indices]  # [num_graphs, output_dim]

        else:
            raise ValueError(f"Unknown node_to_choose: {self.node_to_choose}. Options: mean, sum, last")
        
        return g


class SingleClassifier(nn.Module):
    """
    Minimal linear head for single text classification.
    Supports two variants:
    - Variant 2 (default): hidden_dim -> linear(num_classes) - has separate classifier head
    - Variant 1: encoder already outputs num_classes, no additional linear layer
    """
    def __init__(self, embedder: LayerGINEncoder, hidden_dim: int, num_classes: int, use_variant1: bool = False):
        super().__init__()
        self.enc = embedder
        self.use_variant1 = use_variant1
        
        if use_variant1:
            # Variant 1: Encoder outputs num_classes directly, no additional linear layer
            # Verify encoder was initialized correctly
            if embedder.output_dim != num_classes:
                raise ValueError(f"Variant 1 requires encoder.output_dim ({embedder.output_dim}) == num_classes ({num_classes})")
            self.linear = None
        else:
            # Variant 2: Standard setup with separate linear classifier head
            if embedder.output_dim != hidden_dim:
                raise ValueError(f"Variant 2 requires encoder.output_dim ({embedder.output_dim}) == hidden_dim ({hidden_dim})")
            self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch_graphs: GeomBatch) -> torch.Tensor:
        h = self.enc(batch_graphs)     # [B, H] or [B, C] depending on variant
        if self.use_variant1:
            # Encoder already outputs logits over num_classes
            return h                    # [B, C]
        else:
            # Apply linear classifier head
            return self.linear(h)       # [B, C]



class PairClassifier(nn.Module):
    """Minimal linear head for pair classification."""
    def __init__(self, embedder: LayerGINEncoder, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.enc = embedder
        in_dim = hidden_dim * 4        # h1, h2, |h1-h2|, h1*h2
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, batch_a: GeomBatch, batch_b: GeomBatch) -> torch.Tensor:
        h1 = self.enc(batch_a)         # [B, H]
        h2 = self.enc(batch_b)         # [B, H]
        feats = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=1)
        return self.linear(feats)      # [B, C]



class PairCosineSimScore(nn.Module):
    """Minimal linear head for pair regression tasks (e.g., STS scores)."""
    def __init__(self, embedder: LayerGINEncoder, eps: float = 1e-12):
        super().__init__()
        self.enc = embedder        
        self.eps = eps

    def forward(self, batch_a: GeomBatch, batch_b: GeomBatch) -> torch.Tensor:
        h1 = self.enc(batch_a)         # [B, H]
        h2 = self.enc(batch_b)         # [B, H]
        u = F.normalize(h1, p=2, dim=1, eps=self.eps)
        v = F.normalize(h2, p=2, dim=1, eps=self.eps)
        return (u * v).sum(dim=1)              # cosine similarity in [-1, 1]


class MLPEncoder(nn.Module):
    """
    A small MLP over the chosen input (last/mean/flatten).
    Returns a graph-level embedding H to keep the same interface as GIN.

    Linear mode (use_linear=True):
    - Removes all ReLU activations for fair comparison with linear probing baselines
    - Becomes a stack of linear transformations: Linear -> Dropout -> Linear -> Dropout -> ...
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 use_linear: bool = False):
        super().__init__()
        self.use_linear = use_linear
        layers = []
        if num_layers <= 0:
            layers = [nn.Identity()]
            self.out_dim = in_dim
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if not use_linear:  # Skip ReLU in linear mode
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if not use_linear:  # Skip ReLU in linear mode
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.out_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, batch_or_tuple):
        # batch_or_tuple is (X, y) when used from DataLoader; we only need X here.
        if isinstance(batch_or_tuple, tuple) and len(batch_or_tuple) == 2:
            x, _ = batch_or_tuple
        else:
            x = batch_or_tuple
        return self.net(x)


class SingleClassifierLinearHead(nn.Module):
    """Encoder -> Linear head (logistic-regression-like)."""
    def __init__(self, embedder: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.enc = embedder
        self.linear = nn.Linear(hidden_dim, num_classes)
    def forward(self, batch):
        # MLP path: batch is (X, y) from torch DataLoader
        h = self.enc(batch)              # [B, H]
        return self.linear(h)            # [B, C]


class LearnedWeightingEncoder(nn.Module):
    """
    Learned layer weighting encoder (ELMo-style).

    Learns a scalar weight for each layer and combines via softmax-weighted sum.
    Minimal parameters: only num_layers weights.
    """
    def __init__(self, num_layers: int, layer_dim: int):
        """
        Args:
            num_layers: Number of layers in the model (e.g., 24 for Pythia-410m, 32 for Llama3-8B)
            layer_dim: Dimension of each layer embedding (not used, kept for interface consistency)
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.out_dim = layer_dim  # Output has same dim as input layers

        # Only learnable parameter: one weight per layer
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, batch_or_tuple):
        """
        Args:
            batch_or_tuple: Either (X, y) tuple from DataLoader where X is [B, L, D],
                           or just X as [B, L, D]

        Returns:
            [B, layer_dim] - softmax-weighted combination of layers
        """
        # Handle (X, y) tuple from DataLoader
        if isinstance(batch_or_tuple, tuple) and len(batch_or_tuple) == 2:
            x, _ = batch_or_tuple
        else:
            x = batch_or_tuple

        # x: [batch, num_layers, layer_dim]

        # Normalize weights with softmax (sum to 1)
        weights = F.softmax(self.layer_weights, dim=0)  # [num_layers]

        # Expand weights for broadcasting: [1, num_layers, 1]
        weights = weights.view(1, self.num_layers, 1)

        # Weighted sum: [B, L, D] * [1, L, 1] -> [B, D]
        output = (x * weights).sum(dim=1)  # [B, layer_dim]

        return output


class DeepSetEncoder(nn.Module):
    """
    DeepSet encoder for layer-wise embeddings.

    Architecture: φ(pre-pooling) → POOL → ρ(post-pooling)

    φ: MLP applied to each layer independently (shared weights)
    POOL: Permutation-invariant aggregation (mean or sum)
    ρ: MLP applied to the pooled result

    Linear mode (use_linear=True):
    - Removes all ReLU activations for fair comparison with linear probing baselines
    - Pre and post pooling MLPs become linear transformations
    """
    def __init__(
        self,
        num_layers: int,
        layer_dim: int,
        hidden_dim: int = 256,
        pre_pooling_layers: int = 0,
        post_pooling_layers: int = 0,
        pooling_type: str = "mean",
        dropout: float = 0.1,
        use_linear: bool = False
    ):
        """
        Args:
            num_layers: Number of layers in the input (e.g., 24 for Pythia-410m)
            layer_dim: Dimension of each layer embedding (e.g., 1024)
            hidden_dim: Hidden dimension for MLPs (e.g., 256)
            pre_pooling_layers: Number of MLP layers for φ (0, 1, or 2)
            post_pooling_layers: Number of MLP layers for ρ (0, 1, or 2)
            pooling_type: "mean" or "sum"
            dropout: Dropout rate
            use_linear: If True, remove ReLU activations from MLPs
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.pre_pooling_layers = pre_pooling_layers
        self.post_pooling_layers = post_pooling_layers
        self.pooling_type = pooling_type
        self.dropout_rate = dropout
        self.use_linear = use_linear

        # Pre-pooling MLP (φ): applied to each layer independently
        if pre_pooling_layers > 0:
            layers = []
            layers.append(nn.Linear(layer_dim, hidden_dim))
            if not use_linear:  # Skip ReLU in linear mode
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(pre_pooling_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if not use_linear:  # Skip ReLU in linear mode
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.pre_pool_mlp = nn.Sequential(*layers)
            pre_pool_out_dim = hidden_dim
        else:
            self.pre_pool_mlp = None
            pre_pool_out_dim = layer_dim

        # Post-pooling MLP (ρ): applied to the pooled result
        if post_pooling_layers > 0:
            layers = []
            layers.append(nn.Linear(pre_pool_out_dim, hidden_dim))
            if not use_linear:  # Skip ReLU in linear mode
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(post_pooling_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if not use_linear:  # Skip ReLU in linear mode
                    layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.post_pool_mlp = nn.Sequential(*layers)
            self.out_dim = hidden_dim
        else:
            self.post_pool_mlp = None
            self.out_dim = pre_pool_out_dim

    def forward(self, batch_or_tuple):
        """
        Args:
            batch_or_tuple: Either (X, y) tuple from DataLoader where X is [B, L, D],
                           or just X as [B, L, D]

        Returns:
            [B, out_dim] - DeepSet output
        """
        # Handle (X, y) tuple from DataLoader
        if isinstance(batch_or_tuple, tuple) and len(batch_or_tuple) == 2:
            x, _ = batch_or_tuple
        else:
            x = batch_or_tuple

        # x: [batch, num_layers, layer_dim]
        batch_size = x.size(0)

        # Pre-pooling: apply φ to each layer independently
        if self.pre_pool_mlp is not None:
            # Reshape to [B*L, D] to apply MLP to each layer
            x_flat = x.view(batch_size * self.num_layers, self.layer_dim)
            x_transformed = self.pre_pool_mlp(x_flat)  # [B*L, H]
            # Reshape back to [B, L, H]
            x = x_transformed.view(batch_size, self.num_layers, -1)

        # Pooling: aggregate across layers
        if self.pooling_type == "mean":
            pooled = x.mean(dim=1)  # [B, H or D]
        elif self.pooling_type == "sum":
            pooled = x.sum(dim=1)  # [B, H or D]
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}. Use 'mean' or 'sum'.")

        # Post-pooling: apply ρ to the pooled result
        if self.post_pool_mlp is not None:
            output = self.post_pool_mlp(pooled)  # [B, out_dim]
        else:
            output = pooled  # [B, out_dim]

        return output


class DWAttEncoder(nn.Module):
    """
    Depth-Wise Attention (DWAtt) encoder from ElNokrashy et al. 2024.

    "Depth-Wise Attention (DWAtt): A Layer Fusion Method for Data-Efficient Classification"
    https://arxiv.org/abs/2209.15168

    Paper equations implemented:
        Eq. 10: k_n = PE(n) = W_K · k_pos_n     (keys from positional embeddings)
        Eq. 11: f_n(x) = W_n · LN(gelu(U_n·x))  (MLP with bottleneck structure)
        Eq. 4:  v_n = LN_n(f_V_n(z_n))          (values with per-layer MLP + LayerNorm)
        Eq. 5:  q = 1 + elu(z_L + f_Q(z_L))     (query from last layer only)
        Eq. 7:  Score(q, K) = softmax_n(q·K^T)  (attention scores)
        Eq. 8:  Attend = Score(q, K) · V        (weighted sum of values)
        Eq. 9:  h = z_L + Attend(q, K, V)       (output with residual)

    Key difference from standard attention:
        - Keys are STATIC (learned positional embeddings, don't depend on input)
        - Query comes from LAST layer only (not all positions)
        - Values have per-layer MLPs (not shared weights)

    Two modes:
        1. Paper-faithful (hidden_dim=None): Operates entirely in layer_dim
        2. Controlled comparison (hidden_dim=256): Projects to hidden_dim first,
           operates in that space, for fair comparison with GIN/MLP

    Paper hyperparameters (Table 4):
        - d_pos = 24 (positional embedding dimension, independent of num_layers)
        - γ_Q = γ_V = 0.5 (bottleneck ratio)
        - Learning rate = 1e-5
    """
    def __init__(
        self,
        num_layers: int,
        layer_dim: int,
        hidden_dim: int = None,
        bottleneck_ratio: float = 0.5,
        pos_embed_dim: int = 24,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_layers: Number of LLM layers (L). E.g., 25 for Pythia-410m, 23 for TinyLlama, 33 for Llama3-8B
            layer_dim: Dimension of each layer embedding (D). E.g., 1024 for Pythia-410m
            hidden_dim: If provided, projects input to this dimension first (for fair comparison with GIN).
                        If None (default), operates in layer_dim (paper-faithful).
            bottleneck_ratio: γ in paper, ratio for MLP bottleneck (default: 0.5)
            pos_embed_dim: d_pos in paper, dimension for positional embeddings (default: 24, matches paper Table 4)
            dropout: Dropout rate for attention weights
        """
        super().__init__()

        # Validate inputs
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if layer_dim <= 0:
            raise ValueError(f"layer_dim must be positive, got {layer_dim}")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive if provided, got {hidden_dim}")
        if pos_embed_dim <= 0:
            raise ValueError(f"pos_embed_dim must be positive, got {pos_embed_dim}")
        if not 0 < bottleneck_ratio <= 1:
            raise ValueError(f"bottleneck_ratio must be in (0, 1], got {bottleneck_ratio}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.num_layers = num_layers
        self.layer_dim = layer_dim
        # If hidden_dim provided, project to that space; otherwise operate in layer_dim
        self.working_dim = hidden_dim if hidden_dim is not None else layer_dim
        self.use_input_projection = hidden_dim is not None
        # Paper uses d_pos = 24 (Table 4) - this is independent of num_layers
        self.pos_embed_dim = pos_embed_dim
        self.bottleneck_dim = int(self.working_dim * bottleneck_ratio)

        if self.bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {self.bottleneck_dim} "
                           f"(working_dim={self.working_dim}, bottleneck_ratio={bottleneck_ratio})")

        # === Optional input projection (for fair comparison with GIN) ===
        if self.use_input_projection:
            self.input_projection = nn.Linear(layer_dim, self.working_dim)
        else:
            self.input_projection = None

        # === Keys (STATIC - don't depend on input) ===
        # k_pos_n ∈ R^d_pos initialized uniformly in [0, 1] (Section 9.2, Eq. 10)
        self.layer_pos_embeddings = nn.Parameter(torch.empty(num_layers, self.pos_embed_dim))
        nn.init.uniform_(self.layer_pos_embeddings, 0, 1)

        # W_K projects positional embeddings to working_dim: k_n = W_K · k_pos_n
        self.key_transform = nn.Linear(self.pos_embed_dim, self.working_dim, bias=False)

        # === Values (input-dependent, per-layer MLPs) ===
        # Eq. 11: f_n(x) = W_n · LN(gelu(U_n · x))
        # Eq. 4:  v_n = LN_n(f_V_n(z_n))
        # So we have: MLP (with internal LN) -> external LN
        self.value_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.working_dim, self.bottleneck_dim),  # U_n: down projection
                nn.GELU(),                                         # gelu activation
                nn.LayerNorm(self.bottleneck_dim),                 # LN inside MLP (Eq. 11)
                nn.Linear(self.bottleneck_dim, self.working_dim),  # W_n: up projection
            )
            for _ in range(num_layers)
        ])
        # External LayerNorm after MLP: v_n = LN_n(f_V_n(z_n)) (Eq. 4)
        self.value_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.working_dim) for _ in range(num_layers)
        ])

        # === Query (from last layer only) ===
        # f_Q has same structure as f_V (Eq. 11)
        # q = 1 + elu(z_L + f_Q(z_L)) (Eq. 5)
        self.query_transform = nn.Sequential(
            nn.Linear(self.working_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(self.bottleneck_dim),
            nn.Linear(self.bottleneck_dim, self.working_dim),
        )

        self.dropout = nn.Dropout(dropout)
        # Output dimension equals working_dim
        self.out_dim = self.working_dim

    def forward(self, batch_or_tuple):
        """
        Forward pass implementing DWAtt (Equations 4-9 from paper).

        Args:
            batch_or_tuple: Either (X, y) tuple or just X
                X shape: [batch_size, num_layers, layer_dim]

        Returns:
            output: [batch_size, working_dim] where working_dim is hidden_dim if provided, else layer_dim

        Raises:
            ValueError: If input dimensions don't match expected
            RuntimeError: If NaN detected in computation
        """
        # Handle (X, y) tuple from DataLoader
        if isinstance(batch_or_tuple, tuple) and len(batch_or_tuple) == 2:
            x, _ = batch_or_tuple
        else:
            x = batch_or_tuple

        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, L, D], got shape {x.shape}")
        batch_size, input_layers, input_dim = x.shape
        if input_layers != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {input_layers}")
        if input_dim != self.layer_dim:
            raise ValueError(f"Expected layer_dim={self.layer_dim}, got {input_dim}")

        # === Optional input projection (for fair comparison with GIN) ===
        if self.use_input_projection:
            # Project all layers: [B, L, layer_dim] -> [B, L, working_dim]
            x = self.input_projection(x)

        # Get last layer representation: z_L [B, working_dim]
        z_L = x[:, -1, :]

        # === Compute Keys (STATIC - same for all inputs) ===
        # Eq. 10: k_n = W_K · k_pos_n
        # keys: [L, working_dim]
        keys = self.key_transform(self.layer_pos_embeddings)

        # === Compute Values (input-dependent) ===
        # Eq. 4: v_n = LN_n(f_V_n(z_n))
        # Eq. 11: f_V_n(z) = W_n · LN(gelu(U_n · z))
        values = []
        for n in range(self.num_layers):
            z_n = x[:, n, :]                          # [B, working_dim]
            v_n = self.value_transforms[n](z_n)       # [B, working_dim] - MLP with internal LN
            v_n = self.value_layer_norms[n](v_n)      # [B, working_dim] - external LN
            values.append(v_n)
        values = torch.stack(values, dim=1)           # [B, L, working_dim]

        # === Compute Query (from last layer only) ===
        # Eq. 5: q = 1 + elu(z_L + f_Q(z_L))
        q_transform = self.query_transform(z_L)       # [B, working_dim]
        query = 1 + nn.functional.elu(z_L + q_transform)  # [B, working_dim]

        # === Attention ===
        # Eq. 7: Score(q, K) = softmax_n(q · K^T)
        attn_scores = torch.matmul(query, keys.t())   # [B, working_dim] @ [working_dim, L] -> [B, L]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L]

        # Check for NaN in attention weights (can happen with extreme values)
        if torch.isnan(attn_weights).any():
            raise RuntimeError("NaN detected in attention weights. Check for numerical instability.")

        attn_weights = self.dropout(attn_weights)     # [B, L]

        # Eq. 8: Attend = Score(q, K) · V
        # [B, 1, L] @ [B, L, working_dim] -> [B, 1, working_dim] -> [B, working_dim]
        attended = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)

        # === Output with residual ===
        # Eq. 9: h = z_L + Attend(q, K, V)
        output = z_L + attended                       # [B, working_dim]

        # Check for NaN in output
        if torch.isnan(output).any():
            raise RuntimeError("NaN detected in DWAtt output. Check for numerical instability.")

        return output

    def get_attention_weights(self, x):
        """
        Get attention weights for visualization/analysis.

        Args:
            x: Input tensor [B, L, layer_dim]

        Returns:
            attn_weights: [B, L] - attention weights over layers
        """
        if isinstance(x, tuple):
            x, _ = x

        # Apply input projection if enabled
        if self.use_input_projection:
            x = self.input_projection(x)

        z_L = x[:, -1, :]
        keys = self.key_transform(self.layer_pos_embeddings)
        q_transform = self.query_transform(z_L)
        query = 1 + nn.functional.elu(z_L + q_transform)
        attn_scores = torch.matmul(query, keys.t())
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return attn_weights
