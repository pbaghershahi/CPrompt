import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from model import GCN, LinkPredictionPrompt
from copy import deepcopy
from data_utils import make_datasets
from utils import *


s_dataset, t_dataset = make_datasets(
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
ph_dim = 64
o_dim = 64
n_layers = 2
enc_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
main_model = GCN(t_dataset.n_feats, h_dim, nclass=t_dataset.num_gclass, dropout=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pmodel = LinkPredictionPrompt(
    t_dataset.n_feats,
    h_dim, t_dataset.n_feats,
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
batch_size = 32
n_augs = 2
visualize = False
if visualize:
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(s_dataset.y.unique().size(0))])
else:
    colors = None

ug_graphs = []
losses = []
main_losses = []
main_accs = []
enc_model.eval()
main_model.eval()

with torch.no_grad():
    main_model.eval()
    test_loss, test_acc = test(
        main_model, s_dataset, 32, device, -1, visualize, colors, "main")
    print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

with torch.no_grad():
    main_model.eval()
    test_loss, test_acc = test(
        main_model, t_dataset, 32, device, -1, visualize, colors, "main")
    print(f'Main Loss on Pretrained GNN: {test_loss:.4f}, Main ACC: {test_acc:.3f}')

for epoch in range(n_epochs):
    with torch.no_grad():
        pmodel.eval()
        main_model.eval()
        test_loss, test_acc = test(
            main_model, t_dataset, 32, device, -1, visualize, colors, "prompt")
        print(f'Main Loss: {test_loss:.4f}, Main ACC: {test_acc:.3f}', "#"*100)

    pmodel.train()
    main_model.eval()
    total_loss = 0
    counter = 0
    x_mean = 0
    counter = 0
    for i, batch in enumerate(t_dataset.train_loader):
        optimizer.zero_grad()
        # print(f"Train batch: {i}/{t_dataset.train_idx}")
        batch = batch.to_data_list()
        prompt_batch = [
            aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, mode="drop").to(device)
            for g in batch
        ]
        pos_batch = [
            aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, mode="drop").to(device)
            for g in batch
        ]
        neg_batch = [
            aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, mode="add").to(device)
            for g in batch
        ]
        prompt_batch = pmodel(prompt_batch)

        prompt_x_adj = batch_to_xadj_list(prompt_batch, device)
        pos_x_adj = batch_to_xadj_list(pos_batch, device)
        neg_x_adj = batch_to_xadj_list(neg_batch, device)
        prompt_out, _ = main_model(prompt_x_adj)
        pos_out, _ = main_model(pos_x_adj)
        neg_out, _ = main_model(neg_x_adj)
        # loss = multiclass_triplet_loss(prompt_out, pos_out, neg_out, 1)
        loss = ntxent_loss(prompt_out, pos_out, neg_out)
        loss.backward()
        optimizer.step()
        # total_grad_norm = 0
        # for name, param in pmodel.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             total_grad_norm += param.grad.norm()
        # print("Gradients norm: ", total_grad_norm.item())