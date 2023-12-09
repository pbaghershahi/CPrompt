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
from utils import test, load_model, drop_edges, normalize_, contrastive_loss, glist_to_gbatch
from model import GNNGraphClass, PromptGraph
from copy import deepcopy

# dataset = TUDataset(root='data/TUDataset', name='PROTEINS_full', use_node_attr=True)
# graph_data = TUDataset(root='data/TUDataset', name='MUTAG')
dataset = TUDataset(
    root='data/TUDataset',
    name='PROTEINS_full',
    )

for g in dataset:
    pass
for g in dataset._data_list:
    g.edge_attr = torch.ones((g.edge_index.size(1),), dtype=torch.float)

# torch.manual_seed(432)
dataset = dataset.shuffle()
normalize_(dataset.x)

n_train = int(len(dataset)*0.9)
train_dataset = dataset[:n_train]
test_dataset = dataset[n_train:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# torch.manual_seed(43002)
h_dim = 64
ph_dim = 64
o_dim = 64
n_layers = 2
n_pnodes = 5

enc_model = GNNGraphClass(dataset.x.size(1), h_dim, output_dim=dataset.num_classes, num_layers=n_layers, normalize=True, has_head=False)
main_model = GNNGraphClass(dataset.x.size(1), h_dim, output_dim=dataset.num_classes, num_layers=n_layers, normalize=True, has_head=True)
pmodel = PromptGraph(n_pnodes, dataset.x.size(1), ph_dim, o_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enc_model.to(device)
main_model.to(device)
pmodel.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pmodel.parameters(), lr=1e-2)
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
batch_size = 64
n_augs = 2
n_prom = n_augs//2

aug_graphs = []
losses = []
main_losses = []
main_accs = []
enc_model.eval()
main_model.eval()

with torch.no_grad():
    g_list = []
    for i, g in enumerate(test_dataset):
        g_list.append(g.to(device))
    org_loader = DataLoader(g_list, batch_size=64, shuffle=True)
    main_loss, m_counter = 0, 0
    for org_data in org_loader:
        emb_out = main_model(org_data.x, org_data.edge_index, org_data.edge_attr, org_data.batch)
        main_loss += obj_fun(emb_out, org_data.y)
        m_counter += 1
    main_mean_loss = main_loss/m_counter
    main_losses.append(main_mean_loss)
    train_acc = test(org_loader, main_model)
    main_accs.append(train_acc)
    print(f'Main Loss: {main_mean_loss:.4f}, Main ACC: {train_acc:.3f}')


for epoch in range(n_epochs):
    with torch.no_grad():
        pmodel.eval()
        g_list = []
        for i, g in enumerate(test_dataset):
            g_list.append(g.to(device))
        _ = pmodel(g_list)
        org_loader = DataLoader(g_list, batch_size=64, shuffle=True)
        main_loss, m_counter = 0, 0
        for org_data in org_loader:
            emb_out = main_model(org_data.x, org_data.edge_index, org_data.edge_attr, org_data.batch)
            main_loss += obj_fun(emb_out, org_data.y)
            m_counter += 1
        main_mean_loss = main_loss/m_counter
        main_losses.append(main_mean_loss)
        train_acc = test(org_loader, main_model)
        main_accs.append(train_acc)
        print(f'Main Test Loss: {main_mean_loss:.4f}, Main Test ACC: {train_acc:.3f}')

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
        pbatch_gt = glist_to_gbatch(gbatch_list)
        pbatch_gt = to_dense_adj(pbatch_gt.edge_index, pbatch_gt.batch)
        prompt_graphs = [drop_edges(deepcopy(g), n_drops) for g in gbatch_list]
        _ = pmodel(prompt_graphs)
        pbatch_pred = glist_to_gbatch(prompt_graphs)
        pbatch_pred = to_dense_adj(pbatch_pred.edge_index, pbatch_pred.batch)
        adj_loss = F.binary_cross_entropy_with_logits(pbatch_pred, pbatch_gt)
        gbatch_list = prompt_graphs + gbatch_list
        g_batch = glist_to_gbatch(gbatch_list)
        emb_out = enc_model(g_batch.x, g_batch.edge_index, g_batch.edge_attr, g_batch.batch)
        loss = contrastive_loss(emb_out, temperature, device) + 0.05 * adj_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(epoch, loss.data)
        # total_loss += loss
        # counter += 1
        # if i%batch_size == 0 or i==len(train_dataset):
        #     mean_loss = total_loss/counter
        #     mean_loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     print(epoch, mean_loss.data)
        #     total_loss, counter = 0, 0