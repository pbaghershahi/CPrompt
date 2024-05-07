import torch, random, os
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from utils import *
from data_utils import *
from model import *
from copy import deepcopy
from scipy.spatial.distance import cdist

def pretrain_model(
    s_dataset,
    model_name, 
    model_config,
    optimizer_config,
    training_config,
    eval_step = 1,
    save_model = True, 
    pretext_task = "classification",
    model_dir = "./pretrained"
):
    task = "multi" if s_dataset.num_gclass > 2 else "binary"
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    
    if model_name == "GCN":
        model = PretrainedModel(**model_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if pretext_task == "classification":
        obj_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    test_loss, test_acc, test_f1 = test(model, s_dataset, device, task = task, mode = "pretrain")
    logger.info(f'GNN Before Pretraining -- Loss: {test_loss:.4f} -- ACC: {test_acc:.3f} -- F1-score: {test_f1:.3f}')

    n_epochs = training_config["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
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
            if i % max(1, int((s_dataset.n_train//scores.size(0))*0.2)) == 0:
                logger.info(f"Train batch: {i}/{np.ceil(s_dataset.n_train//scores.size(0))} -- Train Loss: {loss.item()}")
        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0:
            test_loss, test_acc, test_f1 = test(model, s_dataset, device, task = task, mode = "pretrain")
            logger.info(
                "#"*10 + " " +
                f"Epoch: {epoch}/{n_epochs} -- Train Loss: {loss:.4f} -- " +
                f"Main Loss: {test_loss:.4f} -- Main ACC: {test_acc:.3f} -- Main F1: {test_f1:.3f}" +
                " " + "#"*10
            )
    
    if save_model:
        empty_directory(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}_Pretrained_{exec_name}.pth")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path
        )
    else:
        model_path = "Won't be stored"
    return model_path


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
        # ps_probs = torch.as_tensor(np.eye(prompt_out.size(1))[pred_label, :], device = device)
        ps_probs = F.softmax(torch.as_tensor(1 - dd, device = device), dim=1)
        loss = self.obj_fun(prompt_out, ps_probs)
        # softmax_out = F.softmax(prompt_out, dim=1)
        # loss += entropy_loss(softmax_out).mean()
        # b_softmax = softmax_out.mean(dim=0)
        # loss += torch.sum(b_softmax * torch.log(b_softmax + 1e-5))
        return loss

    def all_in_one(self, pretrained_model, prompt_model, batch, device):
        labels = batch.y.to(device)
        prompt_batch = prompt_model(batch, device)
        prompt_out, _ = pretrained_model(
            prompt_batch,
            decoder = True,
            )
        loss = self.obj_fun(prompt_out, labels, reduction="mean")
        return loss
        
    def train(self, t_dataset, pretrained_model, prompt_model, optimizer, device, logger) -> None:
        for i, batch in enumerate(t_dataset.train_loader):
            optimizer.zero_grad()
            loss = self.train_func(pretrained_model, prompt_model, batch, device)
            loss.backward()
            optimizer.step()
            total_grad_norm = 0
            if i % max(1, int((t_dataset.n_train//batch.y.size(0))*0.2)) == 0:
                logger.info(f"Train batch: {i}/{np.ceil(t_dataset.n_train//batch.y.size(0))}, Train Loss: {loss.data}")


def prompting(
    t_dataset,
    prompt_method, 
    prompt_config,
    pretrained_config,
    optimizer_config,
    pretrained_path,
    training_config,
    s_dataset = None,
    num_runs = 5,
    eval_step = 1
):
    task = "multi" if t_dataset.num_gclass > 2 else "binary"
    overal_acc = []
    for _ in range(num_runs):
        exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')
        log_file_path = "./log/"+exec_name+".log"
        logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    
        main_model = PretrainedModel(**pretrained_config)
        if prompt_method == "all_in_one":
            pmodel = HeavyPrompt(**prompt_config)
        elif prompt_method == "contrastive":
            pmodel = BasePrompt(**prompt_config)
        elif prompt_method == "pseudo_labeling":
            pmodel = BasePrompt(**prompt_config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main_model.to(device)
        pmodel.to(device)
        load_model(main_model, read_checkpoint=True, pretrained_path=pretrained_path)
        for param in main_model.parameters():
            param.requires_grad = False
        main_model.eval()
    
        optimizer = Adam(pmodel.parameters(), lr = optimizer_config["lr"])
        scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
        Trainer = PromptTrainer(prompt_method, training_config)
    
        if s_dataset is not None:
            test_loss, test_acc, test_f1 = test(main_model, s_dataset, device, task = task, mode = "pretrain")
            logger.info(f'Pretrained GNN on Source Dataset -- Loss: {test_loss:.4f} -- ACC: {test_acc:.3f} -- F1-score: {test_f1:.3f}')
        test_loss, test_acc, test_f1 = test(main_model, t_dataset, device, task = task, mode = "pretrain")
        logger.info(f'Pretrained GNN on Target Dataset Without Prompting -- Loss: {test_loss:.4f} -- ACC: {test_acc:.3f} -- F1-score: {test_f1:.3f}')
    
        test_average_acc = []
        n_epochs = training_config["n_epochs"]
        for epoch in range(n_epochs):
            pmodel.train()
            main_model.eval()
            Trainer.train(t_dataset, main_model, pmodel, optimizer, device, logger)
            scheduler.step()
            optimizer.zero_grad()
            
            if epoch % eval_step == 0:
                pmodel.eval()
                main_model.eval()
                test_loss, test_acc, test_f1 = test(main_model, t_dataset, device, task = task, mode = "prompt", pmodel = pmodel)
                logger.info(
                    "#"*10 + " " +
                    f"Epoch: {epoch}/{n_epochs} -- Main Loss: {test_loss:.4f} -- " +
                    f"Main ACC: {test_acc:.3f} -- Main F1: {test_f1:.3f}" +
                    " " + "#"*10
                )
                if epoch >= 125:
                    test_average_acc.append(test_acc)
                    
        test_average_acc = np.array(test_average_acc).mean()
        logger.info(f"The average accuracy on test data is: {test_average_acc}")
        overal_acc.append(test_average_acc)
    overal_acc = np.array(overal_acc).mean()
    logger.info(f"Total after {num_runs} runs: {np.array(overal_acc).mean()}")