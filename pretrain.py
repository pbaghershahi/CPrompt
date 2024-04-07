import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from model import GCN, LinkPredictionPrompt
from copy import deepcopy
from datetime import datetime
from data_utils import make_datasets
from utils import *


s_dataset, t_dataset =  make_datasets(
    num_nsamples = 1000,
    num_nclass = 7,
    num_gclass = 5,
    n_feats = 32,
    ng_perclass = 100,
    nn_perclass = (50, 100),
    nlabel_perm = 0.5,
    graph_selec_noise = 0.5,
    cov_scale = 2,
    train_per = 0.85,
    test_per = 0.15,
    norm_mode = "max",
    visualize = False
)

h_dim = 64
n_layers = 2
batch_size = 64
visualize = False
if visualize:
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(s_dataset.y.unique().size(0))])
else:
    colors = None

model = GCN(s_dataset.n_feats, h_dim, nclass=s_dataset.num_gclass, dropout=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# for d in train_loader:
#     d.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

n_epochs = 250
for epoch in range(n_epochs):
    model.train()
    # s_dataset._shuffle()
    for i, batch in enumerate(s_dataset.train_loader):
        print(f"Train batch: {i}/{s_dataset.train_idx}", end='\r')
        # train_batch, train_labels = s_dataset[i:min(i+batch_size, s_dataset.train_idx)]
        x_adj_list = batch_to_xadj_list(batch.to_data_list(), device)
        optimizer.zero_grad()
        emb_out, _ = model(x_adj_list)
        loss = obj_fun(emb_out, batch.y.to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    with torch.no_grad():
        test_loss, test_acc = test(
            model, s_dataset, device, epoch, visualize, colors, "pretrain")
        print(f'Epoch: {epoch}/{n_epochs}, Train Loss: {loss:.4f}, Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

!rm -f /content/CPrompt/pretrained/*
save_model = True
if save_model:
    model_dir = './pretrained'
    os.makedirs(model_dir, exist_ok=True)
    name = datetime.today().strftime('%Y_%m_%d_%H_%M')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_dir, f'model_{name}.pt')
    )