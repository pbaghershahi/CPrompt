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
from torch_geometric.data import Data, Batch
from torchvision.transforms.functional import normalize
from torch_geometric.utils import augmentation, to_dense_adj
from torch_geometric.nn import GAT, GATConv, GCNConv
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import global_mean_pool
from utils import *
import random
from model import GCN, LinkPredictionPrompt
from copy import deepcopy

# dataset = TUDataset(root='data/TUDataset', name='PROTEINS_full', use_node_attr=True)
# graph_data = TUDataset(root='data/TUDataset', name='MUTAG')
dataset = TUDataset(
    root='data/TUDataset',
    name='ENZYMES',
    use_node_attr=True
    )
# dataset = TUDataset(
#     root='data/TUDataset',
#     name='PROTEINS_full',
#     use_node_attr=True
#     )

for g in dataset:
    g.x = normalize_(g.x, mode="max")

torch.manual_seed(2411)
dataset = dataset.shuffle()

n_train = int(len(dataset)*0.9)
train_dataset = dataset[:n_train]
test_dataset = dataset[n_train:]

h_dim = 64
ph_dim = 64
o_dim = 64
n_layers = 2

enc_model = GCN(dataset.x.size(1), h_dim, nclass=dataset.num_classes, dropout=0.2)
main_model = GCN(dataset.x.size(1), h_dim, nclass=dataset.num_classes, dropout=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pmodel = LinkPredictionPrompt(
    dataset.x.size(1),
    h_dim, o_dim,
    num_layers=2,
    normalize=True,
    has_head=False,
    device=device
)
enc_model.to(device)
main_model.to(device)
pmodel.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pmodel.parameters(), lr=1e-3)
pfiles_path = [pfile for pfile in os.listdir("/content/CPrompt/pretrained/") if pfile.endswith(".pt")]
prepath = os.path.join("/content/CPrompt/pretrained/", pfiles_path[0])
load_model(enc_model, read_checkpoint=True, pretrained_path=prepath)
load_model(main_model, read_checkpoint=True, pretrained_path=prepath)
for param in enc_model.parameters():
    param.requires_grad = False
for param in main_model.parameters():
    param.requires_grad = False


n_epochs = 256
temperature = 1
n_drops = 0.15
batch_size = 16
n_augs = 2
visualize = True
if visualize:
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(dataset.y.unique().size(0))])
else:
    colors = None

aug_graphs = []
losses = []
main_losses = []
main_accs = []
enc_model.eval()
main_model.eval()

with torch.no_grad():
    main_model.eval()
    test_loss, test_acc = test(
        main_model, test_dataset, len(test_dataset), device, -1, visualize, colors)
    print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

for epoch in range(n_epochs):
    with torch.no_grad():
        pmodel.eval()
        main_model.eval()
        g_list = [g.to(device) for g in test_dataset]
        labels = [g.y for g in test_dataset]
        p_x_adj = pmodel(g_list)
        test_loss, test_acc = test_prompt(
            main_model, p_x_adj, torch.as_tensor(labels), epoch, visualize, colors)
        print(f'Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}', "#"*100)

    pmodel.train()
    total_loss = 0
    counter = 0
    optimizer.zero_grad()
    train_dataset = train_dataset.shuffle()
    for i in range(0, len(train_dataset), batch_size):
        gbatch_list = []
        for j in range(i, min(i+batch_size, len(train_dataset))):
            g = train_dataset[j].to(device)
            gbatch_list.append(g)
        # pbatch_gt = glist_to_gbatch(gbatch_list)
        # pbatch_gt = to_dense_adj(pbatch_gt.edge_index, pbatch_gt.batch)
        prompt_graphs = [drop_edges(deepcopy(g), n_drops) for g in gbatch_list]
        p_x_adj = pmodel(prompt_graphs)
        g_x_adj = batch_to_xadj_list(gbatch_list, device)
        # print(g_batch.y)
        emb_out1, _ = enc_model(p_x_adj)
        emb_out2, _ = enc_model(g_x_adj)
        # loss = contrastive_loss(emb_out1, emb_out2, temperature, device) + 0.5 * adj_loss
        loss = multiclass_marginal_loss(emb_out1, emb_out2, 1)
        # print("loss: ", loss.item())
        loss.backward()
        # total_norm = 0
        # for name, p in pmodel.named_parameters():
        #     if p.grad is not None and p.requires_grad:
        #         # print("temp norm: ", name, p.grad)
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        # print(loss, total_norm)
        optimizer.step()
        optimizer.zero_grad()