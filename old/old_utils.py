import torch, os
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def glist_to_gbatch(graph_list):
    g_loader = DataLoader(graph_list, batch_size=len(graph_list))
    return next(iter(g_loader))

def normalize_(input_tensor, dim=0):
    mean = input_tensor.mean(dim=dim)
    std = input_tensor.std(dim=dim)
    std = torch.max(std, torch.ones_like(std)*1e-12)
    input_tensor.sub_(mean).div_(std)
    return input_tensor

def test(loader, model):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.max(dim=1)[1]
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

def load_model(cmodel, pmodel=None, read_checkpoint=True, pretrained_path=None):
    if read_checkpoint and pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path)["model_state_dict"]
    else:
        pretrained_dict = pmodel.state_dict()
    model_dict = cmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    cmodel.load_state_dict(pretrained_dict)

def drop_edges(org_graph, drop_prob):
    n_edges = org_graph.edge_index.size(1)
    perm = torch.randperm(n_edges)[int(n_edges * drop_prob):]
    org_graph.edge_index = org_graph.edge_index[:, perm]
    return org_graph

def simmatToadj(adjacency_matrix):
    adjacency_matrix = torch.where(adjacency_matrix >= 0.5)
    adjacency_matrix = torch.cat(
        (adjacency_matrix[0][None, :], adjacency_matrix[1][None, :]),
        dim=0)
    return adjacency_matrix

"""
Contrastive loss between two augmented graphs of one original graph 
with other graphs of a batch.
"""
# def contrastive_loss(emb_mat1, emb_mat2, n_prompt, temperature=1, device='cpu'):
#     sim_mat = emb_mat1 @ emb_mat2.T
#     s_mask = (~(torch.diag(torch.ones((n_prompt)), diagonal=sim_mat.size(1)-n_prompt)[:n_prompt, :]).bool()).float()
#     s_mask = s_mask.to(device)
#     sim_mat *= s_mask
#     sim_mat = sim_mat.flip(dims=(0,))
#     pos_scores = sim_mat.flip(dims=(0,)).diagonal(offset=sim_mat.size(1)-n_prompt)
#     neg_scores = torch.logsumexp(sim_mat, dim=1)
#     loss_partial = neg_scores - pos_scores
#     loss = torch.mean(loss_partial)
#     return loss

"""
Contrastive loss between a graph and its augmentation for all 
original graphs with other graphs of a batch.
"""
def contrastive_loss(emb_mat, temperature=1, device='cpu'):
    sim_mat = emb_mat[:emb_mat.size(0)//2, :] @ emb_mat[emb_mat.size(0)//2:, :].T
    pos_scores = sim_mat.diagonal()
    neg_scores = torch.logsumexp(sim_mat, dim=1)
    loss_partial = neg_scores - pos_scores
    loss = torch.mean(loss_partial)
    return loss