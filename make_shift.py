from torch_geometric.datasets import QM9, TUDataset, CitationFull, Planetoid, Airports
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score
from torch_geometric.nn import GCNConv, GCN
from utils import *
import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class GCNNodeClassification(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=2,
                 dropout=0.5, with_bn=False):
        super(GCNNodeClassification, self).__init__()
        self.layers = nn.ModuleList([GCNConv(num_features, hidden_channels)])
        if with_bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        self.num_layers = num_layers
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.head = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.with_bn = with_bn

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.bns[i](x) if self.with_bn else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        scores = self.head(x)
        return scores, x 

def train(model, data, num_epochs, lr=0.0001):
    optimizer = Adam(model.parameters(), lr)
    train_loss, val_loss, test_loss = [], [], []
    train_accuracy, val_accuracy, test_accuracy = [], [], []
    macro_f1_train, macro_f1_val, macro_f1_test = [], [], []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out, embeds = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            with torch.no_grad():
                train_loss.append(loss.item())
                val_loss.append(F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item())
                test_loss.append(F.cross_entropy(out[data.test_mask], data.y[data.test_mask]).item())
                train_preds = torch.argmax(out[data.train_mask], axis=1)
                val_preds = torch.argmax(out[data.val_mask], axis=1)
                test_preds = torch.argmax(out[data.test_mask], axis=1)
                train_accuracy.append(((train_preds == data.y[data.train_mask]).sum()/len(data.y[data.train_mask])).item())
                val_accuracy.append(((val_preds == data.y[data.val_mask]).sum()/len(data.y[data.val_mask])).item())
                test_accuracy.append(((test_preds == data.y[data.test_mask]).sum()/len(data.y[data.test_mask])).item())
                macro_f1_train.append(f1(train_preds, data.y[data.train_mask].cpu()))
                macro_f1_val.append(f1(val_preds, data.y[data.val_mask].cpu()))
                macro_f1_test.append(f1(test_preds, data.y[data.test_mask].cpu()))
                print(f'Epoch : {epoch:.3f}, Training Accuracy: {train_accuracy[-1]:.3f}, Testing Accuracy: {test_accuracy[-1]:.3f}')
    return out, embeds


class DomianShift():
    def __init__(self, n_domains=2, *args, **kwargs) -> None:
        self.n_domains = n_domains

    def init_splits(self, dataset, embeds, n_domain_sample):
        labels = dataset.y
        if n_domain_sample is None:
            n_domain_sample = labels.size(0)
        self.clusters = kmeans(labels, embeds, self.n_domains)
        self.all_masks = herding(dataset, embeds, self.clusters, n_domain_sample)
        domain_idx_dict = {d:[] for d in range(self.n_domains)}
        for key, value in self.all_masks.items():
            class_mask = (labels == key).nonzero().T.squeeze()
            for d in range(self.n_domains):
                domain_idx_dict[d].append(class_mask[value == d])
        self.domain_idx_dict = {key:torch.cat(value, dim=0).sort().values for key, value in domain_idx_dict.items()}

    def get_splits(self, dataset, embeds, n_domain_sample = None):
        self.init_splits(dataset, embeds, n_domain_sample)
        return self.domain_idx_dict

    def save_to_file(self, dataset_name, dir_path=None):
        if dir_path is None:
            dir_path = "./domains"
        os.makedirs(dir_path, exist_ok=True)
        files_path = os.path.join(dir_path, f"{dataset_name}_domain_idicies.txt")
        with open(files_path, "w") as f_handle:
            for domain_id, idxs in self.domain_idx_dict.items():
                line = f"{domain_id}: "
                line += " ".join(idxs.cpu().numpy().astype(str).tolist())
                f_handle.write(line + "\n")
        return files_path

    @classmethod
    def load_from_file(cls, filename_):
        with open(filename_, "r") as f_handle:
            domain_idx_dict = dict()
            for line in f_handle.readlines():
                domain_id, idxs = line.strip().split(":")
                domain_idx_dict[int(domain_id)] = torch.as_tensor(np.array(idxs.strip(" ").split(" ")).astype(np.int64))
        return domain_idx_dict

def kmeans(labels, embeds, nclass_cluster):
    all_centers = []
    num_class_dict = {i:nclass_cluster for i in labels.unique()}
    print(f"Kmeans for {labels.size(0)} samples and {labels.unique()} classes started!")
    for class_id, cnt in num_class_dict.items():
        class_mask = (labels == class_id)
        num_clusters = min(cnt, class_mask.sum())
        features = embeds[class_mask, :].detach().cpu().numpy()
        k_means = KMeans(n_clusters = num_clusters)
        k_means.fit(features)
        all_centers.append(k_means.cluster_centers_[None, :, :])
    all_centers = np.concatenate(all_centers, axis=0)
    return torch.as_tensor(all_centers)


def herding(dataset, embeds, centers, n_domain_sample):
    n_centers = centers.size(1)
    num_class_dict = {i:n_domain_sample for i in range(dataset.num_classes)}
    all_class_masks = {}
    labels = dataset.y
    print(f"Herding for {labels.size(0)} samples and {dataset.num_classes} classes started!")
    for class_id, cnt in num_class_dict.items():
        class_mask = (labels == class_id)
        class_idxs = torch.where(class_mask)[0]
        features = embeds[class_idxs]
        select_mask = torch.ones((class_idxs.size(0),))* -1
        all_idx_mask = torch.arange(class_idxs.size(0))
        n_unselected = min(class_idxs.size(0), 2*cnt)
        i = 0
        print(f"Will selelct {n_unselected} samples from class {class_id}")
        pbar = tqdm(total = n_unselected)
        while n_unselected >= n_centers:
            for j in range(n_centers):
                residual = centers[class_id, j]*(i+1) - features[select_mask == j].sum(dim=0)
                idx_mask = all_idx_mask[select_mask == -1]
                dists = torch.cdist(residual[None, :], features[idx_mask])
                idx_min = idx_mask[dists.argmin()]
                select_mask[idx_min] = j
                n_unselected -= 1
                pbar.update(1)
                i += 1
        all_class_masks[class_id] = select_mask
    return all_class_masks


def main(args):
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    t_ds_name = args.t_dataset if args.t_dataset != args.s_dataset else args.s_dataset
    logger.info(f"Source Dataset: {args.s_dataset}, Target Dataset: {t_ds_name}. Training, Test: {args.train_per}, {args.test_per} -- Batch size: {args.batch_size}")
    if args.config_from_file == "":
        if args.config_to_file != "":
            with open(args.config_to_file, 'w') as outfile:
                yaml.dump(vars(args), outfile, default_flow_style=False)
            logger.info(f"Config saved to file: {args.config_to_file}")
    else:
        logger.info(f"Reading config from: {args.config_from_file}")
        with open(args.config_from_file, 'r') as infile:
            args = yaml.safe_load(infile)
            args = argparse.Namespace(**args)
            
    if args.s_dataset in ["Cora", "CiteSeer", "PubMed"]:
        s_dataset, t_dataset = get_node_dataset(
            args.s_dataset,
            cov_scale = args.noise_cov_scale,
            mean_shift = args.noise_mean_shift,
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            n_hopes = 2,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed,
            n_split = [140, 250, 1000],
            domain_idxs_dir = "./domains"
        )
    elif args.s_dataset in ["ENZYMES", "PROTEINS_full"]:
        s_dataset, t_dataset = get_graph_dataset(
            args.s_dataset,
            cov_scale = args.noise_cov_scale,
            mean_shift = args.noise_mean_shift,
            store_to_path = "./data/TUDataset",
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed
        )
    elif args.s_dataset in ["Letter-high", "Letter-low", "Letter-med"]:
        s_dataset, t_dataset = get_pyggda_dataset(
            args.s_dataset,
            t_ds_name,
            store_to_path = "./data/TUDataset",
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            norm_mode = "max",
            node_attributes = True,
            seed = args.seed
        )
    elif args.s_dataset in ["digg", "oag", "twitter", "weibo"]:
        s_dataset, t_dataset = get_gda_dataset(
            ds_dir = "./data/ego_network/",
            s_ds_name = args.s_dataset,
            t_ds_name = t_ds_name,
            train_per = args.train_per,
            test_per = args.test_per,
            batch_size = args.batch_size,
            get_s_dataset = True,
            get_t_dataset = True,
            seed = args.seed
        )

    model_name = "GCN"
    model_config = dict(
        d_feat = s_dataset.n_feats,
        d_hid = args.pretrain_h_dim,
        d_class = s_dataset.num_gclass,
        n_layers = 2, r_dropout = 0.2
    )
    optimizer_config = dict(
        lr = args.pretrain_lr,
        scheduler_step_size = args.pretrain_step_size,
        scheduler_gamma = args.pretrain_lr
    )
    training_config = dict(
        n_epochs = args.pretrain_n_epochs
    )
    logger.info(f"Setting for pretraining: Model: {model_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")

    if args.pretrain:
        logger.info(f"Pretraining {model_name} on {args.s_dataset} started for {args.pretrain_n_epochs} epochs")
        _, pretrained_path = pretrain_model(
        s_dataset,
        # t_dataset,
        model_name,
        model_config,
        optimizer_config,
        training_config,
        logger,
        eval_step = args.pretrain_eval_step,
        save_model = args.save_pretrained,
        pretext_task = "classification",
        model_dir = "./pretrained",
        empty_pretrained_dir = args.empty_pretrained_dir
        )
        logger.info(f"Pretraining is finished! Saved to: {pretrained_path}")
    else:
        pretrained_path = args.pretrain_path
        logger.info(f"Using previous pretrained model at {pretrained_path}")

    pretrained_config = model_config
    num_tokens = int(np.ceil(cal_avg_num_nodes(t_dataset)))
    logger.info(f"The number of tokens added: {num_tokens}")
    optimizer_config = dict(
        lr = args.lr,
        scheduler_step_size = args.step_size,
        scheduler_gamma = args.gamma
    )
    logger.info(f"Prompting method: {args.prompt_method} -- Setting: Prompting function: {args.prompt_fn} -- Target Dataset: {t_ds_name}")
    if args.prompt_method == "all_in_one":
        prompt_config = dict(
            token_dim = t_dataset.n_feats,
            token_num = num_tokens,
            cross_prune = args.cross_prune,
            inner_prune = args.inner_prune,
            trans_x = args.trans_x
        )
        training_config = dict(
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Transform X: {args.trans_x} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    elif args.prompt_method == "contrastive":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = args.h_dim,
            output_dim = t_dataset.n_feats,
            prompt_fn = args.prompt_fn,
            token_num = num_tokens
        )
        training_config = dict(
            aug_type = args.aug_type,
            pos_aug_mode = args.pos_aug_mode,
            neg_aug_mode = args.neg_aug_mode,
            p_raug = args.p_raug,
            n_raug = args.n_raug,
            add_link_loss = args.add_link_loss,
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Augmentation type: {args.aug_type} -- Positive aug mode and rate: {args.pos_aug_mode}, {args.p_raug} -- Negative aug mode and rate: {args.neg_aug_mode}, {args.n_raug} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    elif args.prompt_method == "pseudo_labeling":
        prompt_config = dict(
            emb_dim = t_dataset.n_feats,
            h_dim = args.h_dim,
            output_dim = t_dataset.n_feats,
            prompt_fn = args.prompt_fn,
            token_num = num_tokens
        )
        training_config = dict(
            aug_type = args.aug_type,
            pos_aug_mode = args.pos_aug_mode,
            p_raug = args.p_raug,
            n_epochs = args.n_epochs,
            r_reg = args.r_reg
        )
        logger.info(f"Prompt tuning setting: Augmentation type: {args.aug_type} -- Positive aug mode and rate: {args.pos_aug_mode}, {args.p_raug} -- Regularization: {args.r_reg} -- Epochs: {args.n_epochs}")
    else:
        raise Exception("The chosen method is not valid!")

    logger.info(f"Setting for prompt tuning: Prompt: {prompt_config} -- Pretrained Model: {pretrained_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")
    logger.info(f"Prompt tuning started: Num runs: {args.num_runs} -- Eval step: {args.eval_step}")
    pmodel = prompting(
        t_dataset,
        args.prompt_method,
        prompt_config,
        pretrained_config,
        optimizer_config,
        pretrained_path,
        training_config,
        logger,
        s_dataset,
        num_runs = args.num_runs,
        eval_step = args.eval_step
    )


def get_node_dataset(
    ds_name,
    cov_scale = 2,
    mean_shift = 0,
    train_per = 0.85,
    test_per = 0.15,
    batch_size = 32,
    n_hopes = 2,
    norm_mode = "max",
    node_attributes = True,
    seed = 2411,
    n_split = [140, 250, 1000],
    domain_idxs_dir = "./domains"
):

    """ Currently supported datasets:
        - Cora_
    """
    dataset = Planetoid(
        root = f'data/{ds_name}',
        name = ds_name
        )

    ntotal_graphs = dataset._data.size(0)
    if domain_idxs_dir is None:
        model_cora = GCNNodeClassification(
            num_features = dataset.x.size(1), 
            hidden_channels = 256, 
            num_classes = dataset.num_classes, 
            num_layers = 2, 
            dropout = 0.5, 
            with_bn = False
        )
        _, embeds = train(model_cora, dataset._data, 5, lr=0.0001)
        domain_shift = DomianShift()
        domain_idx_dict = domain_shift.get_splits(dataset, embeds)
        domain_shift.save_to_file(ds_name)
    else:
        file_path = os.path.join(domain_idxs_dir, f"{ds_name}_domain_idicies.txt")
        domain_idx_dict = DomianShift.load_from_file(file_path)

    n_train, n_valid, n_test = n_split
    unique_labels = dataset.y.unique()
    n_train_class = n_train//unique_labels.size(0)
    domain_train_idxs = {d_id:[] for d_id in domain_idx_dict.keys()}
    for class_id in unique_labels:
        class_idxs = (dataset.y == class_id).nonzero()
        for d_id, d_idxs in domain_idx_dict.items():
            retained_idxs = torch.isin(d_idxs, (dataset.y == class_id).nonzero().T[0]).nonzero().T[0]
            class_idxs = d_idxs[retained_idxs]
            perm = torch.randperm(class_idxs.size(0))
            domain_train_idxs[d_id].append(class_idxs[perm][:n_train_class])
    domain_train_idxs = {d_id: torch.cat(mask, dim=0).sort().values for d_id, mask in domain_train_idxs.items()}
    domain_valid_idxs = dict()
    domain_test_idxs = dict()
    for d_id, train_idx in domain_train_idxs.items():
        other_mask = ~torch.isin(domain_idx_dict[d_id], train_idx)
        other_idxs = domain_idx_dict[d_id][other_mask.nonzero().T[0]]
        perm = torch.randperm(other_idxs.size(0))
        other_idxs = other_idxs[perm]
        domain_valid_idxs[d_id] = other_idxs[:n_valid].sort().values
        domain_test_idxs[d_id] = other_idxs[n_valid:n_valid + n_test].sort().values

    s_idxs = torch.cat([
        domain_train_idxs[0],
        domain_valid_idxs[0],
        domain_test_idxs[0],
        ])
    t_idxs = torch.cat([
        domain_train_idxs[1],
        domain_valid_idxs[1],
        domain_test_idxs[1],
        ])
    s_data = deepcopy(dataset._data.subgraph(s_idxs))
    t_data = deepcopy(dataset._data.subgraph(t_idxs))
    s_dataset = NodeToGraphDataset(
        s_data,
        n_hopes = n_hopes
        )
    s_n_train = domain_train_idxs[0].size(0)
    s_n_valid = domain_valid_idxs[0].size(0)
    s_n_test = domain_test_idxs[0].size(0)
    s_dataset.initialize(
        train_idxs = torch.arange(s_n_train),
        valid_idxs = torch.arange(s_n_train, s_n_train + s_n_valid),
        test_idxs = torch.arange(s_n_train + s_n_valid, s_n_train + s_n_valid + s_n_test),
        batch_size = batch_size,
        normalize_mode = norm_mode,
        shuffle = False
        )
    t_dataset = NodeToGraphDataset(
        t_data,
        n_hopes = n_hopes
        )
    t_n_train = domain_train_idxs[0].size(0)
    t_n_valid = domain_valid_idxs[0].size(0)
    t_n_test = domain_test_idxs[0].size(0)
    t_dataset.initialize(
        train_idxs = torch.arange(t_n_train),
        valid_idxs = torch.arange(t_n_train, t_n_train + t_n_valid),
        test_idxs = torch.arange(t_n_train + t_n_valid, t_n_train + t_n_valid + t_n_test),
        batch_size = batch_size,
        normalize_mode = norm_mode,
        shuffle = False
        )
    
    return s_dataset, t_dataset


ds_name = "Cora"
cov_scale = 2,
mean_shift = 0,
train_per = 0.85,
test_per = 0.15,
batch_size = 32,
n_hopes = 2,
norm_mode = "max",
node_attributes = True,
seed = 2411

""" Currently supported datasets:
    - Cora_
"""
dataset = Planetoid(
    root = f'data/{ds_name}',
    name = ds_name
    )
ntotal_graphs = dataset._data.size(0)
train_idxs = torch.nonzero(dataset._data.train_mask).view(-1)
valid_idxs = torch.nonzero(dataset._data.val_mask).view(-1)
test_idxs = torch.nonzero(dataset._data.test_mask).view(-1)
other_mask = ~dataset._data.train_mask.logical_or(dataset._data.val_mask).logical_or(dataset._data.test_mask)
task = "multi" if dataset.num_classes > 2 else "binary"
f1 = BinaryF1Score() if task == "binary" else MulticlassF1Score(num_classes=dataset.num_classes, average="macro")
other_idxs = torch.nonzero(other_mask).view(-1)
n_train = train_idxs.size(0)
n_val = valid_idxs.size(0)
n_test = test_idxs.size(0)
n_other = other_idxs.size(0)
fix_seed(seed)
model_cora = GCNNodeClassification(
    num_features = dataset.x.size(1), 
    hidden_channels = 256, 
    num_classes = dataset.num_classes, 
    num_layers = 2, 
    dropout = 0.5, 
    with_bn = False
)
out, embeds = train(model_cora, dataset._data, 150, lr=0.0001)
domain_shift = DomianShift()
domains_idxs = domain_shift.get_splits(dataset, embeds)