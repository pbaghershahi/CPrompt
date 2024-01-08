import torch, os
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from typing import List
from torch_geometric.utils import to_torch_coo_tensor, to_dense_adj


def multiclass_marginal_loss(T1, T2, margin):
    exp_dims = (T1.size(0), T2.size(0))
    T1 = T1.unsqueeze(0)
    T2 = T2.unsqueeze(1)
    T1 = T1.tile(exp_dims[1], 1, 1)
    T2 = T2.tile(1, exp_dims[0], 1)
    dists = (T1 - T2).norm(p=2, dim=2)
    pos_dists = dists.diagonal()
    neg_dists = ((margin - dists).fill_diagonal_(0.))
    neg_dists = torch.max(neg_dists, torch.zeros_like(neg_dists))
    # print("neg_dists: ", neg_dists)
    neg_dists = neg_dists.mean(dim=1)
    loss = (pos_dists + neg_dists).mean()
    return loss

def dense_to_sparse(spmat):
    indices = spmat.nonzero(as_tuple=False)
    spmat = torch.sparse_coo_tensor(
        indices.T,
        spmat[indices[:, 0], indices[:, 1]],
        size=spmat.size(),
        requires_grad=True
        )
    return spmat

def batch_to_xadj_list(g_batch, device):
    x_adj_list = []
    for i, g in enumerate(g_batch):
        g = g.to(device)
        x = g.x
        adj_dense = to_dense_adj(g.edge_index, max_num_nodes=x.size(0)).squeeze()
        deg_mat = adj_dense.sum(dim=1)
        deg_mat_inv = torch.max(deg_mat, torch.ones_like(deg_mat)*1e-6).pow(-1)
        adj_dense = deg_mat_inv.diag() @ adj_dense
        adj_dense.fill_diagonal_(1.)
        adj_sparse = dense_to_sparse(adj_dense)
        x_adj_list.append((x, adj_sparse))
    return x_adj_list

def visualize_and_save_tsne(features_tensor, colors, epoch, save_dir='feature_plots'):
    os.makedirs(save_dir, exist_ok=True)
    if len(features_tensor.shape) != 2:
        raise ValueError("Input tensor should be 2D (batch_size x feature_dim)")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_tensor)
    plt.figure()
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Feature Vectors')
    save_path = os.path.join(save_dir, f'tsne_plot_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

def test(model, dataset, batch_size, device, epoch=None, visualize=False, colors=None):
    model.eval()
    test_loss, correct = 0, 0
    for i in range(0, len(dataset), batch_size):
        test_batch = dataset[i:min(i+batch_size, len(dataset))]
        x_adj_list = batch_to_xadj_list(test_batch, device)
        out, embeds = model(x_adj_list)
        if visualize:
            visualize_and_save_tsne(
                embeds.detach().numpy(), 
                colors[test_batch.y] if colors is not None else None, 
                epoch if epoch is not None else 0)
        test_loss += F.cross_entropy(out, test_batch.y, reduction="sum")
        out = F.softmax(out, dim=1)
        pred = out.max(dim=1)[1]
        correct += int((pred == test_batch.y).sum())
    test_loss /= len(dataset)
    test_acc = correct / len(dataset)
    return test_loss, test_acc

def test_prompt(model, x_adj_batch, labels, epoch=None, visualize=False, colors=None):
    test_out, embeds = model(x_adj_batch)
    if visualize:
        visualize_and_save_tsne(
            embeds.detach().numpy(), 
            colors[labels] if colors is not None else None, 
            epoch if epoch is not None else 0)
    test_loss = F.cross_entropy(test_out, labels, reduction="mean")
    test_out = F.softmax(test_out, dim=1)
    test_pred = test_out.max(dim=1)[1]
    test_acc = int((test_pred == labels).sum()) / len(x_adj_batch)
    return test_loss, test_acc

def glist_to_gbatch(graph_list):
    g_loader = DataLoader(graph_list, batch_size=len(graph_list))
    return next(iter(g_loader))

def normalize_(input_tensor, dim=0, mode="max"):
    if mode == "max":
        max_value = input_tensor.max(dim=0).values
        input_tensor.div_(torch.max(max_value, torch.ones_like(max_value)*1e-6))
    else:
        mean = input_tensor.mean(dim=dim)
        std = input_tensor.std(dim=dim)
        std = torch.max(std, torch.ones_like(std)*1e-12)
        input_tensor.sub_(mean).div_(std)
    return input_tensor

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