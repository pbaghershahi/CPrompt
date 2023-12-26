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
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from utils import simmatToadj, dense_to_sparse


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


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.head = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x_adj_list):
        g_embeds = []
        for i, (x, adj) in enumerate(x_adj_list):
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            # x = self.gc2(x, adj)
            g_embeds.append(x.mean(dim=0))
            # return F.log_softmax(x, dim=1)
        g_ambeds = torch.stack(g_embeds)
        scores = self.head(g_ambeds)
        return scores


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x, adj):
        support = torch.mm(input_x, self.weight)
        # print("self.weight: ", self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LinkPredictionPrompt(nn.Module):
    def __init__(self,
                 emb_dim,
                 h_dim,
                 output_dim,
                 num_layers=2,
                 normalize=False,
                 has_head=True,
                 device="cuda:0") -> None:
        super(LinkPredictionPrompt, self).__init__()
        self.device = torch.device(device)
        self.gnn1 = GCNConv(emb_dim, h_dim)
        self.bn1 = nn.BatchNorm1d(h_dim)
        self.gnn2 = GCNConv(h_dim, h_dim)
        self.bn2 = nn.BatchNorm1d(h_dim)
        self.head = nn.Linear(h_dim, output_dim)

    def simmatToadj(self, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.triu(diagonal=1)
        adj_list = torch.where(adjacency_matrix >= 0.5)
        edge_weights = adjacency_matrix[adj_list]
        adj_list = torch.cat(
            (adj_list[0][None, :], adj_list[1][None, :]),
            dim=0)
        return adj_list, edge_weights

    def forward_gnn(self, graph):
        emb_out = graph.x.squeeze()
        # print("second graph edge_index, graph edge_attr: ", graph.edge_index.size(), graph.edge_attr.size())
        emb_out = self.gnn1(emb_out, graph.edge_index)
        emb_out = F.relu(emb_out)
        emb_out = self.gnn2(emb_out, graph.edge_index)
        return emb_out

    def forward(self, graphs):
        x_adj_list = []
        for graph in graphs:
            org_adj_mat = to_dense_adj(
                graph.edge_index,
                max_num_nodes=graph.x.size(0)
                ).squeeze()
            emb_out = self.forward_gnn(graph)
            adj_scores = emb_out @ emb_out.T
            pred_adj_mat = F.sigmoid(adj_scores)
            # print("pred_adj_mat: ", pred_adj_mat)
            pred_adj_mat = pred_adj_mat.masked_fill(org_adj_mat.bool(), 1.)
            pred_adj_mat = pred_adj_mat.where(pred_adj_mat >= 0.5, 0.)
            deg_mat = pred_adj_mat.sum(dim=1)
            deg_mat_inv = torch.max(deg_mat, torch.ones_like(deg_mat)*1e-6).pow(-1)
            pred_adj_mat = deg_mat_inv.diag() @ pred_adj_mat
            pred_adj_mat.fill_diagonal_(1.)
            pred_adj_mat = dense_to_sparse(pred_adj_mat)
            x_adj_list.append((graph.x, pred_adj_mat))
        return x_adj_list