import torch
import numpy as np
from typing import List, Any
from torch_geometric.nn import MessagePassing
from .text_automodel_wrapper import TextLayerwiseAutoModelWrapper, TextModelSpecifications

class Aggregator(MessagePassing):
    def __init__(self, aggr_type: str = 'mean'):
        super().__init__(aggr=aggr_type)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class MultiAggregator(torch.nn.Module):
    def __init__(self, num_layers: int = 1, aggr_type: str = 'mean'):
        super().__init__()
        # create N copies of your simple aggregator
        self.layers = torch.nn.ModuleList([
            Aggregator(aggr_type) for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
        return x


class SimpleAggregationWrapper(TextLayerwiseAutoModelWrapper):
    """
    A wrapper that applies simple aggregation operations to layer-wise embeddings.
    No learnable parameters - just aggregation.
    """
    def __init__(self, 
                 model_specs: TextModelSpecifications, 
                 device_map="auto", 
                 evaluation_layer_idx: int = -1,
                 aggregation_method: str = "mean",
                 token_pooling_method: str = "mean",
                 num_gnn_layers: int = 1,
                 nodes_pooling_method: str = "last",
                 graph_type: str = "fully_connected",
                 cayley_jumps: List[int] = [1, 3]):
        super().__init__(model_specs, device_map, evaluation_layer_idx)

        # Validation
        if aggregation_method not in {"add", "mean", "max"}:
            # how gnn uses the neighbors embeddings
            raise ValueError(
                f"Unknown aggregation method: {aggregation_method}. Use 'add', 'mean', or 'max'"
            )
        if token_pooling_method not in {"mean", "first_hidden_state", "last_hidden_state"}:
            # Per sample we need to decide how to pool the tokens: first, last, mean
            raise ValueError(
                f"Unknown pooling method: {token_pooling_method}. Use 'mean', 'first_hidden_state', or 'last_hidden_state'"
            )
        if graph_type not in {"linear", "fully_connected", "virtual_node", "cayley"}:
            # decide type of graph to use
            raise ValueError(
                f"Unknown graph type: {graph_type}. Use 'linear', 'fully_connected', 'virtual_node', or 'cayley'"
            )
        if nodes_pooling_method not in {"last", "first", "mean"}:
            # decide which node to use after message passing: last, first, mean
            raise ValueError(
                f"Unknown nodes pooling method: {nodes_pooling_method}. Use 'last', 'first', or 'mean'"
            )

        self.aggregation_method = aggregation_method
        self.token_pooling_method = token_pooling_method
        self.nodes_pooling_method = nodes_pooling_method
        self.num_gnn_layers = num_gnn_layers
        self.graph_type = graph_type
        
        if self.graph_type == "cayley":
            self.cayley_jumps = [1, 3]
        else:
            self.cayley_jumps = []
        
        self.aggregator = MultiAggregator(
            num_layers=self.num_gnn_layers,
            aggr_type=self.aggregation_method,
        )

    def _build_edge_index(self, num_nodes: int) -> torch.LongTensor:
        # returns [2, E] tensor
        if self.graph_type == "linear":
            # 0-1-2-...-(L-1)
            src = torch.arange(num_nodes-1)
            dst = torch.arange(1, num_nodes)
            edge = torch.stack([src, dst], dim=0)
            edge = torch.cat([edge, edge.flip(0)], dim=1)

        elif self.graph_type == "fully_connected":
            # all i<j connected
            comb = torch.combinations(torch.arange(num_nodes), r=2).t()
            edge = torch.cat([comb, comb.flip(0)], dim=1)

        elif self.graph_type == "virtual_node":            
            # num_nodes already includes the virtual node at index num_nodes-1
            virt_idx = num_nodes - 1
            real = torch.arange(virt_idx, device=self.device)          # 0..virt_idx-1
            virt = torch.full((virt_idx,), virt_idx, device=self.device)
            edge = torch.stack([torch.cat([real, virt]),
                                torch.cat([virt, real])], dim=0)

        elif self.graph_type == "cayley":
            src_list, dst_list = [], []
            for j in self.cayley_jumps:
                i = torch.arange(num_nodes, device=self.device)
                src_list.append(i)
                dst_list.append((i + j) % num_nodes)
                src_list.append(i)
                dst_list.append((i - j) % num_nodes)
            edge = torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)

        return edge.to(self.device)
    
    @torch.no_grad()
    def encode(self, input_data: List[str], **kwargs: dict) -> np.ndarray:
        """
        Encode text using the base model, then apply simple aggregation to layer-wise embeddings.
        """
        # Get layer-wise embeddings from the base model
        kwargs["pooling_method"] = self.token_pooling_method
        result = super().encode(input_data, return_raw_hidden_states=True, **kwargs)
        embeddings, raw_sample_hidden_states, layerwise_encodings = result
        # Note: raw_sample_hidden_states may be None when using memory-efficient hooks
        
        # layerwise_encodings shape: (num_layers, num_samples, embedding_dim)
        num_layers, num_samples, embedding_dim = layerwise_encodings.shape
        
        # Apply aggregation to each sample's layer embeddings
        aggregated_embeddings = []
        for sample_idx in range(num_samples):
            # Extract layer embeddings for this sample: (num_layers, embedding_dim)
            sample_layer_embeddings = layerwise_encodings[:, sample_idx, :]
            
            # Convert to torch tensor and move to device
            sample_layer_embeddings = torch.tensor(sample_layer_embeddings, dtype=torch.float, device=self.device)
            
            # Handle virtual node case
            if self.graph_type == "virtual_node":
                # Add virtual node with zero embeddings
                virtual_node_embedding = torch.zeros(embedding_dim, dtype=torch.float, device=self.device)
                sample_layer_embeddings = torch.cat([sample_layer_embeddings, virtual_node_embedding.unsqueeze(0)], dim=0)
            
            # Create graph based on the specified type
            num_layers = sample_layer_embeddings.shape[0]
            edge_index = self._build_edge_index(num_nodes=num_layers)
            
            # Apply aggregation
            with torch.no_grad():
                aggregated_output = self.aggregator(sample_layer_embeddings, edge_index)
                
                # For virtual node, exclude the virtual node from final aggregation
                if self.graph_type == "virtual_node":
                    aggregated_output = aggregated_output[:-1]  # Remove virtual node
                
                if self.nodes_pooling_method == "mean":
                    # Take the mean across all nodes to get a single embedding
                    final_embedding = torch.mean(aggregated_output, dim=0)
                elif self.nodes_pooling_method == "last":
                    final_embedding = aggregated_output[-1]
                elif self.nodes_pooling_method == "first":
                    final_embedding = aggregated_output[0]
                else:
                    raise ValueError(f"Unknown nodes pooling method: {self.nodes_pooling_method}. Use 'last', 'first', or 'mean'")
                aggregated_embeddings.append(final_embedding.cpu().numpy())
        
        return np.array(aggregated_embeddings) 