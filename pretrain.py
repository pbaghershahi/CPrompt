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


# s_dataset, t_dataset = get_graph_dataset(
#     "ENZYMES",
#     cov_scale = 2,
#     mean_shift = 10.,
#     train_per = 0.80,
#     test_per = 0.20,
#     batch_size = 32,
#     norm_mode = "max",
#     node_attributes = True,
#     visualize = False
# )
# task = "multi"

# s_dataset, t_dataset = get_node_dataset(
#     "Cora",
#     cov_scale = 0.1,
#     mean_shift = 0.,
#     train_per = 0.80,
#     test_per = 0.20,
#     batch_size = 32,
#     n_hopes = 2,
#     norm_mode = "max",
#     node_attributes = True,
#     visualize = False
# )
# task = "multi"

# s_dataset, t_dataset = get_gda_dataset(
#     ds_dir = "./data/ego_network/",
#     s_ds_name = "oag",
#     t_ds_name = "digg",
#     train_per = 0.80,
#     test_per = 0.20,
#     batch_size = 4096,
# )
# task = "binary"

# s_dataset, t_dataset = get_pyggda_dataset(
#     "Letter-low",
#     "Letter-high",
#     store_to_path = "./data/TUDataset",
#     train_per = 0.80,
#     test_per = 0.20,
#     batch_size = 32,
#     norm_mode = "max",
#     node_attributes = True,
# )
# task = "multi"

# s_dataset, t_dataset = get_pyg_node_gda_dataset(
#     "USA",
#     # "Brazil",
#     "Europe",
#     store_to_path = "./data/Airports",
#     train_per = 0.80,
#     test_per = 0.20,
#     batch_size = 32,
#     norm_mode = "max",
#     n_hopes = 2,
#     node_attributes = True,
# )
# task = "multi"

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
model = PretrainedModel(
    d_feat = s_dataset.n_feats,
    d_hid = h_dim,
    d_class = s_dataset.num_gclass,
    n_layers = 2,
    r_dropout = 0.2
    )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# for d in train_loader:
#     d.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

with torch.no_grad():
    model.eval()
    # ipdb.set_trace()
    test_loss, test_acc, test_f1 = test(model, s_dataset, device, mode = "pretrain")
    print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}, Main F1: {test_f1:.3f}')

n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    # s_dataset._shuffle()
    for i, batch in enumerate(s_dataset.train_loader):
        optimizer.zero_grad()
        scores, _ = model(
            batch,
            decoder = True,
            device = device
        )
        loss = obj_fun(scores, batch.y.to(device))
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Train batch: {i}/{s_dataset.n_train} | Loss: {loss.item()}")
    scheduler.step()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            test_loss, test_acc, test_f1 = test(model, s_dataset, device, mode = "pretrain")
            print(f'Epoch: {epoch}/{n_epochs}, Train Loss: {loss:.4f}, Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}, Main F1: {test_f1:.3f}')

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