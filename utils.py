import torch, random, os, logging
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.utils import to_dense_adj, subgraph, remove_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score

def setup_logger(
    name,
    level=logging.DEBUG,
    stream_handler=True,
    file_handler=True,
    log_file='default.log'
    ):
    open(log_file, 'w').close()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
        )

    if stream_handler:
        sth = logging.StreamHandler()
        sth.setLevel(level)
        sth.setFormatter(formatter)
        logger.addHandler(sth)

    if file_handler:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def fix_seed(seed_value, random_lib=False, numpy_lib=False, torch_lib=False):
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

def cal_entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-8)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def cal_JSD(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    jsd = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean', log_target=False) \
                 + F.kl_div(q.log(), m, reduction='batchmean', log_target=False))
    return jsd

def get_adj_labels(g_batch):
    adj_matrices = []
    for g in g_batch:
        adj_mat = to_dense_adj(g.edge_index, max_num_nodes=g.x.size(0)).squeeze()
        adj_matrices.append(adj_mat.view(-1))
    adj_matrices = torch.cat(adj_matrices)
    return adj_matrices

def link_predict_loss(graphs, gt_adj_matrix):
    all_scores = []
    for g in graphs:
        x = g.x
        scores = x @ x.T
        all_scores.append(F.sigmoid(scores).view(-1))
    all_scores = torch.cat(all_scores).view(-1)
    loss = F.binary_cross_entropy_with_logits(all_scores, gt_adj_matrix)
    return loss

def scale_array(in_tensor, t_min, t_max):
    min_v = in_tensor.min()
    max_v = in_tensor.max()
    in_tensor = (in_tensor-min_v)*(t_max-t_min)/(max_v-min_v) + t_min
    return in_tensor

def ntxent_loss(
        prompt_score, 
        pos_score, 
        neg_score, 
        labels=None, 
        weighting=None):
    pr_norm = prompt_score.norm(p=2, dim=1)
    pr_norm = torch.max(pr_norm, torch.ones_like(pr_norm)*1e-8)
    p_norm = pos_score.norm(p=2, dim=1)
    p_norm = torch.max(p_norm, torch.ones_like(p_norm)*1e-8)
    pos = (prompt_score * pos_score).sum(dim=1)/(pr_norm * p_norm)
    pr_norm = pr_norm[:, None].tile(1, pr_norm.size(0))
    n_norm = neg_score.norm(p=2, dim=1)
    n_norm = torch.max(n_norm, torch.ones_like(n_norm)*1e-8)
    n_norm = n_norm[None, :].tile(pr_norm.size(0), 1)
    neg = (prompt_score @ neg_score.T) / (pr_norm * n_norm)

    if labels is not None:
        # print("labels are being used!")
        min_neg = labels.size(0) + 1
        neg_labels = torch.zeros((labels.size(0), labels.size(0))).bool()
        for i in range(labels.size(0)):
            neg_labels[i, :] = (labels != labels[i])
        min_neg = neg_labels.sum(dim=0).min()
        idx_mask = torch.arange(neg.size(1)).tile(neg.size(0), 1)
        idx_mask = torch.where(neg_labels, idx_mask, neg.size(1)+1)
        idx_mask = idx_mask.sort(dim=1, descending=False)[0][:, :min_neg]
        neg = neg.gather(1, idx_mask)

    neg = torch.cat([neg, pos[:, None]], dim=1)
    if weighting == "distance":
        # print("Weighted method is being applied!")
        exp_dims = (prompt_score.size(0), pos_score.size(0))
        prompt_score = prompt_score.unsqueeze(0)
        pos_score = pos_score.unsqueeze(1)
        prompt_score = prompt_score.tile(exp_dims[1], 1, 1)
        pos_score = pos_score.tile(1, exp_dims[0], 1)
        pos_dists = (prompt_score - pos_score).norm(p=2, dim=2).T
        pos_dists = pos_dists.diagonal()
        neg_score = neg_score.unsqueeze(1)
        neg_score = neg_score.tile(1, exp_dims[1], 1)
        neg_dists = (prompt_score - neg_score).norm(p=2, dim=2).T
        neg_dists = torch.cat([neg_dists, pos_dists[:, None]], dim=1).detach().clone()
        weights = (neg_dists.min(dim=1).values)/neg_dists[:, -1]
    elif weighting == "similarity":
        dists = (1/neg).detach().clone()
        # min_dists = dists.min(dim=1).values[:, None]
        # weights = min_dists / dists
        weights = (dists.min(dim=1).values)/dists[:, -1]
        pos += weights
        neg[:, -1] = pos
    else:
        pass
    neg = torch.logsumexp(neg, dim=1)
    loss = (-pos + neg).mean()
    return loss

def multiclass_triplet_loss(prompt_score, pos_score, neg_score, margin, labels=None):
    exp_dims = (prompt_score.size(0), pos_score.size(0))
    prompt_score = prompt_score.unsqueeze(0)
    pos_score = pos_score.unsqueeze(1)
    prompt_score = prompt_score.tile(exp_dims[1], 1, 1)
    pos_score = pos_score.tile(1, exp_dims[0], 1)
    pos_dists = (prompt_score - pos_score).norm(p=2, dim=2).T
    pos_dists = pos_dists.diagonal()
    neg_score = neg_score.unsqueeze(1)
    neg_score = neg_score.tile(1, exp_dims[1], 1)
    neg_dists = (prompt_score - neg_score).norm(p=2, dim=2).T
    neg_dists = (margin - neg_dists).fill_diagonal_(0.)

    if labels is not None:
        # print("labels are being used!")
        min_neg = labels.size(0) + 1
        neg_labels = torch.zeros((labels.size(0), labels.size(0))).bool()
        for i in range(labels.size(0)):
            neg_labels[i, :] = (labels != labels[i])
        min_neg = neg_labels.sum(dim=0).min()
        idx_mask = torch.arange(neg_dists.size(1)).tile(neg_dists.size(0), 1)
        idx_mask = torch.where(neg_labels, idx_mask, neg_dists.size(1)+1)
        idx_mask = idx_mask.sort(dim=1, descending=False)[0][:, :min_neg]
        neg_dists = neg_dists.gather(1, idx_mask)

    neg_dists = torch.max(neg_dists, torch.zeros_like(neg_dists))
    neg_dists = neg_dists.mean(dim=1)
    # print("Mean of the positive samples: ", pos_dists.mean().item(), "Mean of negative samples: ", neg_dists.mean().item())
    loss = (pos_dists + neg_dists).mean()
    return loss

def aug_graph(org_graph, aug_prob, aug_type="link", mode="drop"):
    if aug_type == "link":
        edges = org_graph.edge_index
        n_edges = org_graph.edge_index.size(1)
        n_changes = int(n_edges * aug_prob)
        perm = torch.randperm(n_edges)[n_changes:]
        new_edges = edges[:, perm]
        if mode != "mask":
            add_edges = torch.stack(
                [torch.randint(org_graph.x.size(0), (n_changes,)),
                torch.randint(org_graph.x.size(0), (n_changes,))]
                )
            new_edges = torch.cat([new_edges, add_edges], dim=1)
            new_edges = remove_self_loops(coalesce(new_edges))[0]
        org_graph.edge_index = new_edges
    elif aug_type == "feature":
        n_nodes, n_feats = org_graph.x.size()
        x = org_graph.x
        if mode != "mask":
            n_changes = int(n_feats * aug_prob)
            perm_idxs = torch.randperm(n_feats)[:n_changes]
            for idx in perm_idxs:
                temp_perm = torch.randperm(n_nodes)
                x[:, idx] = x[temp_perm, idx]
        else:
            x = x.reshape(-1)
            n_changes = int(x.size(0) * aug_prob)
            perm = torch.randperm(x.size(0))[:n_changes]
            x[perm] = .0
    return org_graph

def test(model, dataset, device, task, mode, pmodel=None):
    model.eval()
    f1 = BinaryF1Score() if task == "binary" else MulticlassF1Score(num_classes=dataset.num_gclass, average="micro")
    test_loss, correct = 0, 0
    labels = []
    preds = []
    for i, batch in enumerate(dataset.test_loader):
        print("Test batch:", "@"*25, f"{i}/{len(dataset.test_loader)}", "@"*25, end='\r')
        temp_labels = batch.y.to(device)
        if mode == "prompt":
            batch = pmodel(batch, device)
        test_out, _ = model(
            batch,
            decoder = True,
            device = device
            )
        test_loss += F.cross_entropy(test_out, temp_labels, reduction="sum")
        test_out = F.softmax(test_out, dim=1)
        labels.append(temp_labels)
        preds.append(test_out.argmax(dim=1))
        batch, temp_labels, test_out = 0, 0, 0
    # ipdb.set_trace()
    labels = torch.cat(labels)
    preds = torch.cat(preds)
    test_loss /= dataset.n_test
    test_acc = int((labels == preds).sum()) / dataset.n_test
    test_f1 = f1(preds.detach().cpu(), labels.detach().cpu())
    return test_loss, test_acc, test_f1

def get_subgraph(graph, node_indices):
    node_indices = np.sort(node_indices)
    org_edges = subgraph(torch.as_tensor(node_indices), graph.edge_index)[0]
    nodes_mapping = dict(zip(node_indices, np.arange(node_indices.shape[0])))
    sub_edges = torch.empty_like(org_edges.view(-1))
    sub_edges[:] = torch.tensor([nodes_mapping[val.item()] for val in org_edges.view(-1)])
    sub_edges = sub_edges.view(2, -1)
    sub_x = graph.x[node_indices]
    sub_y = graph.y[node_indices]
    sub_g = Data(x=sub_x, edge_index=sub_edges, y=sub_y)
    return sub_g

def normalize_(input_tensor, dim=0, mode="max"):
    if mode == "max":
        max_value = input_tensor.max(dim=0).values
        max_value[max_value==0] = 1.
        input_tensor.div_(max_value)
    else:
        mean = input_tensor.mean(dim=dim)
        std = input_tensor.std(dim=dim)
        std = torch.max(std, torch.ones_like(std)*1e-12)
        input_tensor.sub_(mean).div_(std)
    return input_tensor

def add_multivariate_noise(input_feats, mean, cov_matrix, inplace=True):
    n_samples, n_feats = input_feats.numpy().shape
    noise = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    noise = torch.as_tensor(noise, dtype=torch.float32)
    if inplace:
        return input_feats.add_(noise)
    return input_feats + noise

def _dense_to_sparse(spmat):
    indices = spmat.nonzero(as_tuple=False)
    spmat = torch.sparse_coo_tensor(
        indices.T,
        spmat[indices[:, 0], indices[:, 1]],
        size=spmat.size(),
        requires_grad=True
        )
    return spmat

def batch_to_xadj_list(g_batch, device):
    x_adj_list = []
    for i, g in enumerate(g_batch):
        g = g.to(device)
        adj_dense = to_dense_adj(g.edge_index, max_num_nodes=g.x.size(0)).squeeze()
        deg_mat = adj_dense.sum(dim=1)
        deg_mat_inv = torch.max(deg_mat, torch.ones_like(deg_mat)*1e-6).pow(-1)
        deg_mat_inv = deg_mat_inv.diag()
        adj_dense = deg_mat_inv @ adj_dense
        adj_dense.fill_diagonal_(1.)
        adj_sparse = _dense_to_sparse(adj_dense)
        x_adj_list.append((g.x, adj_sparse))
    return x_adj_list

def visualize_and_save_tsne(features_tensor, colors, epoch, save_dir='feature_plots'):
    os.makedirs(save_dir, exist_ok=True)
    if len(features_tensor.shape) != 2:
        raise ValueError("Input tensor should be 2D (batch_size x feature_dim)")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_tensor)
    plt.figure()
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Feature Vectors')
    save_path = os.path.join(save_dir, f'tsne_plot_{epoch}.png')
    plt.savefig(save_path)
    plt.close()

def glist_to_gbatch(graph_list):
    g_loader = DataLoader(graph_list, batch_size=len(graph_list))
    return next(iter(g_loader))

def load_model(cmodel, pmodel=None, read_checkpoint=True, pretrained_path=None):
    if read_checkpoint and pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path)["model_state_dict"]
    else:
        pretrained_dict = pmodel.state_dict()
    model_dict = cmodel.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    cmodel.load_state_dict(pretrained_dict)

def drop_edges(org_graph, drop_prob):
    n_edges = org_graph.edge_index.size(1)
    perm = torch.randperm(n_edges)[int(n_edges * drop_prob):]
    org_graph.edge_index = org_graph.edge_index[:, perm]
    return org_graph

def simmatToadj(adjacency_matrix):
    adjacency_matrix = torch.where(adjacency_matrix >= 0.5)
    adjacency_matrix = torch.cat(
        (adjacency_matrix[0][None, :], adjacency_matrix[1][None, :]),
        dim=0)
    return adjacency_matrix

def empty_directory(directory):
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                empty_directory(filepath)
                os.rmdir(filepath)
        except Exception as e:
            print(f"Failed to delete {filepath}: {e}")

"""
Contrastive loss between two augmented graphs of one original graph
with other graphs of a batch.
"""
# def contrastive_loss(emb_mat1, emb_mat2, n_prompt, temperature=1, device='cpu'):
#     sim_mat = emb_mat1 @ emb_mat2.T
#     s_mask = (~(torch.diag(torch.ones((n_prompt)), diagonal=sim_mat.size(1)-n_prompt)[:n_prompt, :]).bool()).float()
#     s_mask = s_mask.to(device)
#     sim_mat *= s_mask
#     sim_mat = sim_mat.flip(dims=(0,))
#     pos_scores = sim_mat.flip(dims=(0,)).diagonal(offset=sim_mat.size(1)-n_prompt)
#     neg_scores = torch.logsumexp(sim_mat, dim=1)
#     loss_partial = neg_scores - pos_scores
#     loss = torch.mean(loss_partial)
#     return loss

"""
Contrastive loss between a graph and its augmentation for all
original graphs with other graphs of a batch.
"""

# def contrastive_loss(emb_mat, temperature=1, device='cpu'):
#     sim_mat = emb_mat[:emb_mat.size(0)//2, :] @ emb_mat[emb_mat.size(0)//2:, :].T
#     pos_scores = sim_mat.diagonal()
#     neg_scores = torch.logsumexp(sim_mat, dim=1)
#     loss_partial = neg_scores - pos_scores
#     loss = torch.mean(loss_partial)
#     return loss