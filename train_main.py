import torch, os
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from model import GNNGraphClass
from utils import test, load_model, drop_edges, normalize_

# dataset = TUDataset(root='data/TUDataset', name='MUTAG')
# dataset = TUDataset(root='data/TUDataset', name='NCI1')
dataset = TUDataset(root='data/TUDataset', name='PROTEINS_full')


torch.manual_seed(432)
dataset = dataset.shuffle()
normalize_(dataset.x)
n_train = int(len(dataset)*0.9)
train_dataset = dataset[:n_train]
test_dataset = dataset[n_train:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

h_dim = 64
n_layers = 2
n_pnodes = 100

model = GNNGraphClass(dataset.x.size(1), h_dim, output_dim=dataset.num_classes, num_layers=n_layers, normalize=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for d in train_loader:
    d.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        emb_out = model(data.x, data.edge_index, data.batch)
        loss = obj_fun(emb_out, data.y)
        loss.backward()
        optimizer.step()
        # print(loss.data)
    model.to('cpu')
    # train_acc = test(train_loader, model)
    test_acc = test(test_loader, model)
    model.to(device)
    print(f'train loss {loss:.3f}, test accuracy {test_acc:.3f}')

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