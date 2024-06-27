import torch, random, os, ipdb
import torch.nn.functional as F
import numpy as np
from utils import *
from data_utils import *
from model import *
from scipy.spatial.distance import cdist
from scipy.special import softmax


class PromptTrainer():
    def __init__(self, training_method, training_config, *args, **kwargs) -> None:
        super(PromptTrainer, self).__init__()
        self.training_config = training_config
        if training_method == "supervised":
            self.train_func = self.supervised
        elif training_method == "contrastive":
            self.train_func = self.contrastive
        elif training_method == "pseudo_labeling":
            self.train_func = self.pseudo_labeling
        elif training_method == "fix_match":
            self.train_func = self.fix_match

    def contrastive(self, pretrained_model, prompt_model, batch, device, **kwargs):
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
        loss = ntxent_loss(prompt_out, pos_out, neg_out, weighting=None)
        if self.training_config["add_link_loss"]:
            adj_labels = get_adj_labels(batch).to(device)
            loss += link_predict_loss(prompt_batch, adj_labels)
        if self.training_config["r_reg"] > 0.0 and prompt_model.prompt_fn == "add_tokens":
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def pseudo_labeling(self, pretrained_model, prompt_model, batch, device, **kwargs):
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
        clutering_iters = self.training_config["clutering_iters"]
        if self.training_config["iterative_clustering"]:
            pos_embeds = pos_embed.detach().cpu().numpy()
            pos_probs = pos_out.detach().cpu().numpy()
            # ipdb.set_trace()
            ent_div_ratio = self.training_config["entropy_div_ratio"]
            min_entropy_bank = np.zeros((len(batch),), dtype = bool)
            for r in range(clutering_iters):
               # TODO: we can only find the low entropy points and then apply kmeans clustering to find the cluster centers.
               if r == 0:
                   pos_probs = softmax(pos_probs, axis = 1)
               else:
                   pos_probs = softmax(1 - dd, axis = 1)
               unselected_idxs = (~min_entropy_bank).nonzero()[0]
               pos_entropy = cal_entropy(pos_probs[unselected_idxs, :])
               argmin_entropy = pos_entropy.argsort()
               new_idxs = unselected_idxs[argmin_entropy[:int(np.ceil(len(batch)//ent_div_ratio))]]
               min_entropy_bank[new_idxs] = True
               low_entropy_probs = pos_probs[min_entropy_bank, :]
               low_entropy_embed = pos_embeds[min_entropy_bank, :]
               initc = low_entropy_probs.T @ low_entropy_embed
               initc = np.where(initc > 0, initc, 1e-8)
               # initc = initc / (1e-8 + pos_probs.sum(axis=0)[:,None])
               dd = cdist(pos_embeds, initc, 'cosine')
               pred_label = dd.argmin(axis=1)
        else:
            pos_probs = F.softmax(pos_out, dim=1)
            pos_embeds = (pos_embed.T / pos_embed.norm(p=2, dim=1)).T
            ps_embds = pos_embeds.detach().cpu().numpy()
            ps_probs = pos_probs.detach().cpu().numpy()
            initc = ps_probs.T @ ps_embds
            initc = initc / (1e-8 + ps_probs.sum(axis=0)[:,None])
            dd = cdist(ps_embds, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            for _ in range(clutering_iters):
               # TODO: Can't we use the actual probabilities or soft labels rather than hard labels?
               ps_probs = np.eye(ps_probs.shape[1])[pred_label, :]
               initc = ps_probs.T @ ps_embds
               initc = np.where(initc > 0, initc, 1e-8)
               initc = initc / (1e-8 + ps_probs.sum(axis=0)[:,None])
               dd = cdist(ps_embds, initc, 'cosine')
               pred_label = dd.argmin(axis=1)
        # soft_probs = F.softmax(torch.as_tensor(1 - dd, device = device), dim=1)
        # pos_entropy = cal_entropy(soft_probs.detach().cpu().numpy())
        # argmin_entropy = pos_entropy.argsort()
        # pred_label = pred_label[argmin_entropy[:len(batch)//2]]
        # prompt_out = prompt_out[argmin_entropy[:len(batch)//2]]
        if self.training_config["soft_label"]:
            ps_probs = F.softmax(torch.as_tensor(1 - dd, device = device), dim=1)
        else:
            ps_probs = torch.as_tensor(np.eye(prompt_out.size(1))[pred_label, :], device = device)
        # ent_weights = cal_entropy(soft_probs.detach().cpu().numpy())
        # ent_weights = ent_weights/ent_weights.max(axis=0)
        # ent_weights = torch.as_tensor(ent_weights).to(prompt_out.device)[:, None]
        # ps_probs *= ent_weights
        if self.training_config["binary_task"]:
            loss = F.binary_cross_entropy_with_logits(prompt_out, ps_probs)
        else:
            loss = F.cross_entropy(prompt_out, ps_probs)
        softmax_out = F.softmax(prompt_out, dim=1)
        if self.training_config["w_entropy_loss"] > 0.0:
            ent_loss = entropy_loss(softmax_out).mean()
            if self.training_config["w_softmax_loss"] > 0.0:
                b_softmax = softmax_out.mean(dim=0)
                ent_loss -= self.training_config["w_softmax_loss"] * torch.sum(-b_softmax * torch.log(b_softmax + 1e-5))
            loss += ent_loss * self.training_config["w_entropy_loss"]
        if self.training_config["w_domain_loss"] > 0.0:
            loss += self.training_config["w_domain_loss"] * \
            self.domain_loss(kwargs["discriminator"], kwargs["optimizer_d"], pos_embed, prompt_embed, device)
        if self.training_config["r_reg"] > 0.0 and prompt_model.prompt_fn == "add_tokens":
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def fix_match(self, pretrained_model, prompt_model, batch, device, **kwargs):
        batch = batch.to_data_list()
        pos_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["light_aug_prob"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["light_aug_mode"]
            ).to(device)
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
        pos_out = F.softmax(pos_out, dim=1)
        thresh_mask = (pos_out.max(dim=1).values > self.training_config["cut_off"]).nonzero().T[0]
        pos_out = pos_out[thresh_mask, :]
        prompt_out = prompt_out[thresh_mask, :]
        pseudo_labels = torch.zeros_like(pos_out).scatter_(1, pos_out.argmax(dim=1)[:, None], 1.)
        ce_loss, ent_loss, softmax_loss, domain_loss = 0.0, 0.0, 0.0, 0.0
        if self.training_config["binary_task"]:
            ce_loss = F.binary_cross_entropy_with_logits(prompt_out, pseudo_labels)
        else:
            ce_loss = F.cross_entropy(prompt_out, pseudo_labels)
        softmax_out = F.softmax(prompt_out, dim=1)
        if self.training_config["w_entropy_loss"] > 0.0:
            ent_loss = entropy_loss(softmax_out).mean()
        if self.training_config["w_softmax_loss"] > 0.0:
            b_softmax = softmax_out.mean(dim=0)
            softmax_loss = -torch.sum(-b_softmax * torch.log(b_softmax + 1e-5))
        if self.training_config["w_domain_loss"] > 0.0:
            domain_loss = self.domain_loss(kwargs["discriminator"], kwargs["optimizer_d"], pos_embed, prompt_embed, device)
        # if self.training_config["r_reg"] > 0.0 and prompt_model.prompt_fn == "add_tokens":
        #     loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        # print(f"CE Loss: {ce_loss.item()} -- Entropy Loss: {ent_loss} -- Softmax Loss: {softmax_loss.item()} -- Domain Loss: {domain_loss.item()}")
        loss = ce_loss + ent_loss * self.training_config["w_entropy_loss"] + \
        softmax_loss * self.training_config["w_softmax_loss"] + domain_loss * self.training_config["w_domain_loss"]
        return loss

    def supervised(self, pretrained_model, prompt_model, batch, device, **kwargs):
        labels = batch.y.to(device)
        prompt_batch = prompt_model(batch, device)
        prompt_out, _ = pretrained_model(
            prompt_batch,
            decoder = True,
            )
        loss = F.cross_entropy(prompt_out, labels, reduction="mean")
        if self.training_config["r_reg"] > 0.0 and not prompt_model.trans_x:
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def domain_loss(self, discriminator, optimizer_d, pos_embed, prompt_embed, device):

        real_labels = torch.ones((prompt_embed.size(0), 1), device = device)
        fake_labels = torch.zeros((prompt_embed.size(0), 1), device = device)

        optimizer_d.zero_grad()
        real_out = discriminator(pos_embed)
        fake_out = discriminator(prompt_embed.detach())
        real_loss = F.binary_cross_entropy_with_logits(real_out, real_labels)
        fake_loss = F.binary_cross_entropy_with_logits(fake_out, fake_labels)
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        optimizer_d.step()

        prompt_out = discriminator(prompt_embed)
        prompt_loss = F.binary_cross_entropy_with_logits(prompt_out, real_labels)
        return prompt_loss
        
    def train(self, t_dataset, pretrained_model, prompt_model, optimizer, device, logger, **kwargs) -> None:
        for i, batch in enumerate(t_dataset.train_loader):
            optimizer.zero_grad()
            loss = self.train_func(pretrained_model, prompt_model, batch, device, **kwargs)
            loss.backward()
            optimizer.step()
            total_grad_norm = 0
            # if i % max(1, int((t_dataset.n_train//batch.y.size(0))*0.5)) == 0:
            #     logger.info(f"Train batch: {i}/{np.ceil(t_dataset.n_train//batch.y.size(0))}, Train Loss: {loss.data}")
        return loss