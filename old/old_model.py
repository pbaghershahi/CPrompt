import torch, os
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import augmentation
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from utils import simmatToadj


class PromptGraph(nn.Module):
    def __init__(self, num_pnodes, emb_dim) -> None:
        super(PromptGraph, self).__init__()
        self.prompt_x = nn.Parameter(torch.FloatTensor(num_pnodes, emb_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.prompt_x)

    def forward(self, graphs):
        prompt_adj = self.prompt_x @ self.prompt_x.T
        adj_prompt = F.sigmoid(prompt_adj)
        adj_prompt = torch.triu(adj_prompt, diagonal=1)
        adj_prompt = simmatToadj(adj_prompt)
        for graph in graphs:
            cnum_nodes = graph.x.size(0)
            adj_probs = graph.x @ self.prompt_x.T
            adj_matrix = F.sigmoid(adj_probs)
            adj_matrix = simmatToadj(adj_matrix)
            adj_matrix[1, :] += cnum_nodes
            graph.x = torch.cat((graph.x, self.prompt_x), dim=0)
            graph.edge_index = torch.cat((graph.edge_index, adj_matrix, adj_prompt+cnum_nodes), dim=1)
        return graphs

class GNNGraphClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=1, normalize=True, has_head=True) -> None:
        super(GNNGraphClass, self).__init__()
        output_dim = hidden_dim if output_dim is None else output_dim
        in_dims = [input_dim]+[hidden_dim]*(num_layers-1)
        out_dims = [hidden_dim]*(num_layers)
        inner_layers = [GCNConv(in_dim, out_dim) for (in_dim, out_dim) in zip(in_dims, out_dims)]
        self.normalize = normalize
        if normalize:
            inner_layers = [inner_layers[i//2] if i%2==0 else nn.BatchNorm1d(out_dims[i//2]) for i in range(num_layers*2)]
        self.gnn_layers = nn.ModuleList(inner_layers)
        self.has_head = has_head
        if has_head and output_dim is not None:
            self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_x, edge_index, weights=None, batch=None):
        emb_out = graph_x.squeeze().to(torch.float32)

        i = 0
        while i < len(self.gnn_layers):
            if weights is None:
                emb_out = self.gnn_layers[i](emb_out, edge_index)
            else:
                emb_out = self.gnn_layers[i](emb_out, edge_index, weights)
            if self.normalize:
                i += 1
                emb_out = F.relu(self.gnn_layers[i](emb_out))
            i += 1
        emb_out = global_mean_pool(emb_out, batch)
        emb_out = F.dropout(emb_out, p=0.1, training=self.training)
        if self.has_head:
            emb_out = self.head(emb_out)
        return emb_out
    
class LinkPredictionPrompt(nn.Module):
    def __init__(self, num_pnodes, emb_dim, h_dim, out_dim) -> None:
        super(PromptGraph, self).__init__()
        self.p_layer1 = nn.Linear(emb_dim, h_dim)
        self.bn = nn.BatchNorm1d(h_dim)
        self.p_layer2 = nn.Linear(h_dim, out_dim)
        nn.init.kaiming_uniform_(self.p_layer1.weight)
        nn.init.kaiming_uniform_(self.p_layer2.weight)

    def simmatToadj(self, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.triu(diagonal=1)
        adj_list = torch.where(adjacency_matrix >= 0.5)
        edge_weights = adjacency_matrix[adj_list]
        adj_list = torch.cat(
            (adj_list[0][None, :], adj_list[1][None, :]),
            dim=0)
        return adj_list, edge_weights

    def align_adj_weight(self, graph, adj_list, edge_weights):
        edges = graph.edge_index
        org_weights = torch.ones((edges.size(1),), dtype=torch.float)
        new_edges = torch.cat((edges, adj_list), dim=1)
        org_weights = org_weights.to(self.p_layer1.weight.device)
        new_weights = torch.cat((org_weights, edge_weights), dim=0)
        old_ones = org_weights[org_weights==1].sum()
        new_edges = new_edges.T
        aug_edges = new_edges.select(1, 0) * (graph.x.size(0)+1) + new_edges.select(1, 1)
        s_idxs = aug_edges.sort(stable=True).indices
        new_edges = new_edges.index_select(0, s_idxs).T
        new_weights = new_weights[s_idxs]
        new_ones = new_weights[new_weights==1].sum()
        # before_dict = {}
        # for i in range(new_edges.size(1)):
        #     if (new_edges[0, i].item(), new_edges[1, i].item()) not in before_dict:
        #         before_dict[(new_edges[0, i].item(), new_edges[1, i].item())] = new_weights[i].item()
        #     else:
        #         print("this edge was duplicated: ", (new_edges[0, i].item(), new_edges[1, i].item()), new_weights[i].item())
        unique_edges, inverse, counts = new_edges.unique(dim=1, sorted=True, return_inverse=True, return_counts=True)
        unique_counts = counts.cumsum(dim=0).roll(1, dims=0)
        unique_counts[0] = 0
        new_weights = new_weights[unique_counts]
        new_ones = new_weights[new_weights==1].sum()
        # if old_ones.item() != new_ones.item():
        #     print(old_ones.item(), new_ones.item())
        #     raise ValueError('Oooooooops tar.')
        # after_dict = {}
        # for i in range(unique_edges.size(1)):
        #     if (unique_edges[0, i].item(), unique_edges[1, i].item()) not in after_dict:
        #         after_dict[(unique_edges[0, i].item(), unique_edges[1, i].item())] = new_weights[i].item()
        # for e in graph.edge_index.T:
        #     if after_dict[(e[0].item(), e[1].item())] != 1:
        #         print("this is bug: ", (e[0].item(), e[1].item()), after_dict[(e[0].item(), e[1].item())])
        #         raise ValueError('Ooooooopsss!')
        graph.edge_index = unique_edges
        graph.edge_attr = new_weights
        return graph

    def forward(self, graphs):
        for graph in graphs:
            cnum_nodes = graph.x.size(0)
            emb_out = self.p_layer1(graph.x)
            emb_out = self.bn(emb_out)
            emb_out = F.dropout(F.relu(emb_out), p=0.1, training=self.training)
            emb_out = self.p_layer2(emb_out)
            adj_scores = emb_out @ emb_out.T
            adj_matrix = F.sigmoid(adj_scores)
            adj_list, weights = self.simmatToadj(adj_matrix)
            _ = self.align_adj_weight(graph, adj_list, weights)
        return graphs