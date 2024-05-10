import torch, random, os, ipdb
import torch.nn.functional as F
import numpy as np
from utils import *
from data_utils import *
from model import *
from scipy.spatial.distance import cdist

class PromptTrainer():
    def __init__(self, prompt_method, training_config, *args, **kwargs) -> None:
        super(PromptTrainer, self).__init__()
        self.training_config = training_config
        if prompt_method == "all_in_one":
            self.train_func = self.all_in_one
            self.obj_fun = F.cross_entropy
        elif prompt_method == "contrastive":
            self.train_func = self.contrastive
            self.obj_fun = ntxent_loss
        elif prompt_method == "pseudo_labeling":
            self.train_func = self.pseudo_labeling
            self.obj_fun = F.cross_entropy

    def contrastive(self, pretrained_model, prompt_model, batch, device):
        labels = batch.y.to(device)
        batch = batch.to_data_list()
        # prompt_batch = [
        #     aug_graph(Data(x=g.x, edge_index=g.edge_index, y=g.y), n_drops, aug_type = aug_type, mode = aug_mode).to(device)
        #     for g in batch
        # ]
        pos_batch = [
            Data(x = g.x, edge_index = g.edge_index, y = g.y).to(device)
            for g in batch
        ]
        prompt_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["p_raug"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["pos_aug_mode"]
            ).to(device)
            for g in batch
        ]
        neg_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["n_raug"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["neg_aug_mode"]
            ).to(device)
            for g in batch
        ]
        prompt_batch = prompt_model(prompt_batch)
        prompt_out, prompt_embed = pretrained_model(prompt_batch, decoder = True)
        pos_out, pos_embed = pretrained_model(pos_batch, decoder = True)
        neg_out, neg_embed = pretrained_model(neg_batch, decoder = True)
        # loss = multiclass_triplet_loss(prompt_out, pos_out, neg_out, 1)
        loss = self.obj_fun(prompt_out, pos_out, neg_out, weighting=None)
        if self.training_config["add_link_loss"]:
            adj_labels = get_adj_labels(batch).to(device)
            loss += link_predict_loss(prompt_batch, adj_labels)
        if self.training_config["r_reg"] > 0.0 and prompt_model.prompt_fn == "add_tokens":
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def pseudo_labeling(self, pretrained_model, prompt_model, batch, device):
        labels = batch.y.to(device)
        batch = batch.to_data_list()
        pos_batch = [
            Data(x = g.x, edge_index = g.edge_index, y = g.y).to(device)
            for g in batch
        ]
        prompt_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["p_raug"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["pos_aug_mode"]
            ).to(device)
            for g in batch
        ]
        prompt_batch = prompt_model(prompt_batch)
        prompt_out, prompt_embed = pretrained_model(
            prompt_batch,
            decoder = True,
            device = device
            )
        pos_out, pos_embed = pretrained_model(
            pos_batch,
            decoder = True,
            device = device
            )
        pos_probs = F.softmax(pos_out, dim=1)
        pos_embed = (pos_embed.T / pos_embed.norm(p=2, dim=1)).T
        ps_embds = pos_embed.detach().cpu().numpy()
        ps_probs = pos_probs.detach().cpu().numpy()
        initc = ps_probs.T @ ps_embds
        initc = initc / (1e-8 + ps_probs.sum(axis=0)[:,None])
        dd = cdist(ps_embds, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for round in range(2):
            # TODO: Can't we use the actual probabilities or soft labels rather than hard labels?
            ps_probs = np.eye(ps_probs.shape[1])[pred_label, :]
            initc = ps_probs.T @ ps_embds
            initc = np.where(initc > 0, initc, 1e-8)
            initc = initc / (1e-8 + ps_probs.sum(axis=0)[:,None])
            dd = cdist(ps_embds, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
        ps_probs = torch.as_tensor(np.eye(prompt_out.size(1))[pred_label, :], device = device)
        # ps_probs = F.softmax(torch.as_tensor(1 - dd, device = device), dim=1)
        loss = self.obj_fun(prompt_out, ps_probs)
        # softmax_out = F.softmax(prompt_out, dim=1)
        # loss += entropy_loss(softmax_out).mean()
        # b_softmax = softmax_out.mean(dim=0)
        # loss += torch.sum(b_softmax * torch.log(b_softmax + 1e-5))
        if self.training_config["r_reg"] > 0.0 and prompt_model.prompt_fn == "add_tokens":
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def all_in_one(self, pretrained_model, prompt_model, batch, device):
        labels = batch.y.to(device)
        prompt_batch = prompt_model(batch, device)
        prompt_out, _ = pretrained_model(
            prompt_batch,
            decoder = True,
            )
        loss = self.obj_fun(prompt_out, labels, reduction="mean")
        if self.training_config["r_reg"] > 0.0 and not prompt_model.trans_x:
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss
        
    def train(self, t_dataset, pretrained_model, prompt_model, optimizer, device, logger) -> None:
        for i, batch in enumerate(t_dataset.train_loader):
            optimizer.zero_grad()
            loss = self.train_func(pretrained_model, prompt_model, batch, device)
            # ipdb.set_trace()
            # for name, param in prompt_model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            loss.backward()
            optimizer.step()
            total_grad_norm = 0
            if i % max(1, int((t_dataset.n_train//batch.y.size(0))*0.5)) == 0:
                logger.info(f"Train batch: {i}/{np.ceil(t_dataset.n_train//batch.y.size(0))}, Train Loss: {loss.data}")

