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
from model import GCN
from utils import *
import random
from torch_geometric.utils import to_torch_coo_tensor


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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

h_dim = 64
n_layers = 2
batch_size = 64
visualize = True
if visualize:
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(dataset.y.unique().size(0))])
else:
    colors = None

model = GCN(dataset.x.size(1), h_dim, nclass=dataset.num_classes, dropout=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for d in train_loader:
    d.to(device)
obj_fun = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

n_epochs = 300
for epoch in range(n_epochs):
    model.train()
    train_dataset = train_dataset.shuffle()
    for i in range(0, len(train_dataset), batch_size):
        train_batch = train_dataset[i:min(i+batch_size, len(train_dataset))]
        x_adj_list = batch_to_xadj_list(train_batch, device)
        optimizer.zero_grad()
        emb_out, _ = model(x_adj_list)
        loss = obj_fun(emb_out, train_batch.y.to(device))
        loss.backward()
        optimizer.step()
    scheduler.step()
    test_loss, test_acc = test(
        model, test_dataset, len(test_dataset), device, epoch, visualize, colors)
    print(f'Epoch: {epoch}/{n_epochs}, Train Loss: {loss:.4f}, Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

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