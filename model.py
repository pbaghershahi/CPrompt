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
from torch_geometric.nn import GATConv
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
        # prompt_x = torch.tile(self.prompt_x, dims=(input_feats.size(0), 1, 1))
        # feats = torch.cat((input_feats, self.prompt_x), dim=0)
        # adj_probs = torch
        prompt_adj = self.prompt_x @ self.prompt_x.T
        adj_prompt = F.sigmoid(prompt_adj)
        adj_prompt = torch.triu(adj_prompt, diagonal=1)
        adj_prompt = simmatToadj(adj_prompt) + self.prompt_x.size(0)
        for graph in graphs:
            adj_probs = graph.x @ self.prompt_x.T
            adj_matrix = F.sigmoid(adj_probs)
            adj_matrix = simmatToadj(adj_matrix)
            adj_matrix[1, :] += self.prompt_x.size(0)
            graph.x = torch.cat((graph.x, self.prompt_x), dim=0)
            graph.edge_index = torch.cat((graph.edge_index, adj_prompt, adj_matrix), dim=1)
        return graphs

class GNNGraphClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=1, normalize=True, has_head=True) -> None:
        super(GNNGraphClass, self).__init__()
        output_dim = hidden_dim if output_dim is None else output_dim
        in_dims = [input_dim]+[hidden_dim]*(num_layers-1)
        out_dims = [hidden_dim]*(num_layers)
        inner_layers = [GATConv(in_dim, out_dim) for (in_dim, out_dim) in zip(in_dims, out_dims)]
        self.normalize = normalize
        if normalize:
            inner_layers = [inner_layers[i//2] if i%2==0 else nn.BatchNorm1d(out_dims[i//2]) for i in range(num_layers*2)]
        self.gnn_layers = nn.ModuleList(inner_layers)
        self.has_head = has_head
        if has_head and output_dim is not None:
            self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_x, edge_index, batch):
        emb_out = graph_x.squeeze().to(torch.float32)
        i = 0
        while i < len(self.gnn_layers):
            emb_out = self.gnn_layers[i](emb_out, edge_index)
            if self.normalize:
                i += 1
                emb_out = F.relu(self.gnn_layers[i](emb_out))
            i += 1
        emb_out = global_mean_pool(emb_out, batch)
        emb_out = F.dropout(emb_out, p=0.5, training=self.training)
        if self.has_head:
            emb_out = self.head(emb_out)
        return emb_out