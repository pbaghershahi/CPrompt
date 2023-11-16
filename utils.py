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

def test(loader, model):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
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

def contrastive_loss(emb_mat1, emb_mat2, n_prompt, temperature=1, device='cpu'):
    sim_mat = (emb_mat1 @ emb_mat2.T)[None, :].tile(n_prompt, 1)
    # sim_mat /= temperature
    pos_scores = sim_mat.diagonal(offset=n_prompt)
    s_mask = (~(torch.diag(torch.ones((n_prompt)), diagonal=n_prompt)[:n_prompt, :]).bool()).float()
    s_mask = s_mask.to(device)
    sim_mat *= s_mask
    neg_scores = torch.logsumexp(sim_mat, dim=1)
    loss_partial = neg_scores - pos_scores
    loss = torch.mean(loss_partial)
    return loss