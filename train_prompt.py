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
from torch_geometric.datasets import QM9, TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import global_mean_pool
from utils import test, load_model, drop_edges, contrastive_loss
from model import GNNGraphClass, PromptGraph
from copy import deepcopy

graph_data = TUDataset(root='data/TUDataset', name='MUTAG')

torch.manual_seed(432)
dataset = graph_data.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

torch.manual_seed(43002)
h_dim = 64
n_layers = 3
n_pnodes = 5

enc_model = GNNGraphClass(graph_data.x.size(1), h_dim, output_dim=dataset.num_classes, num_layers=n_layers, normalize=True, has_head=False)
main_model = GNNGraphClass(graph_data.x.size(1), h_dim, output_dim=dataset.num_classes, num_layers=n_layers, normalize=True, has_head=True)
pmodel = PromptGraph(n_pnodes, graph_data.x.size(1))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
batch_size = 32
n_augs = 2
n_prom = n_augs//2

train_loader = DataLoader(graph_data, batch_size=64, shuffle=True)
org_emb = []
for data in train_loader:
    data = data.to(device)
    org_emb.append(enc_model(data.x, data.edge_index, data.batch))
org_emb = torch.cat(org_emb, dim=0)

aug_graphs = []
losses = []
main_losses = []
main_accs = []
enc_model.eval()
main_model.eval()

with torch.no_grad():
    g_list = []
    for i, g in enumerate(graph_data):
        g_list.append(g.to(device))
    org_loader = DataLoader(g_list, batch_size=64, shuffle=True)
    main_loss, m_counter = 0, 0
    for org_data in org_loader:
        emb_out = main_model(org_data.x, org_data.edge_index, org_data.batch)
        main_loss += obj_fun(emb_out, org_data.y)
        m_counter += 1
    main_mean_loss = main_loss/m_counter
    # print("Main Loss: ", main_mean_loss)
    main_losses.append(main_mean_loss)
    main_model.to('cpu')
    train_acc = test(org_loader, main_model)
    main_accs.append(train_acc)
    # test_acc = test(test_loader, main_model)
    main_model.to(device)
    print(f'Main Loss: {main_mean_loss:.4f}, Main ACC: {train_acc:.3f}')


for epoch in range(n_epochs):
    with torch.no_grad():
        pmodel.eval()
        g_list = []
        for i, g in enumerate(graph_data):
            g_list.append(g.to(device))
        _ = pmodel(g_list)
        org_loader = DataLoader(g_list, batch_size=64, shuffle=True)
        main_loss, m_counter = 0, 0
        for org_data in org_loader:
            emb_out = main_model(org_data.x, org_data.edge_index, org_data.batch)
            main_loss += obj_fun(emb_out, org_data.y)
            m_counter += 1
        main_mean_loss = main_loss/m_counter
        # print("Main Loss: ", main_mean_loss)
        main_losses.append(main_mean_loss)
        main_model.to('cpu')
        train_acc = test(org_loader, main_model)
        main_accs.append(train_acc)
        # test_acc = test(test_loader, main_model)
        main_model.to(device)
        print(f'Main Loss: {main_mean_loss:.4f}, Main ACC: {train_acc:.3f}')

    pmodel.train()
    total_loss = 0
    counter = 0
    for i, g in enumerate(graph_data):
        g = g.to(device)
        optimizer.zero_grad()
        aug_graphs = []
        for _ in range(n_augs):
            aug_graphs.append(drop_edges(deepcopy(g), n_drops))
        _ = pmodel(aug_graphs[n_prom:])
        # g_batch = Batch.from_data_list(aug_graphs[0])
        g_loader = DataLoader(aug_graphs, batch_size=len(aug_graphs), shuffle=True)
        g_batch = next(iter(g_loader))
        emb_out = enc_model(g_batch.x, g_batch.edge_index, g_batch.batch)
        loss = contrastive_loss(org_emb[i, :], emb_out, n_prom, temperature, device)
        # loss.backward()
        # optimizer.step()
        # losses.append(loss.detach().numpy())
        # print(epoch, loss.data)
        total_loss += loss
        counter += 1
        if i%batch_size == 0 or i==len(graph_data):
            mean_loss = total_loss/counter
            mean_loss.backward()
            optimizer.step()
            losses.append(mean_loss.detach().numpy())
            print(epoch, mean_loss.data)
            total_loss, counter = 0, 0