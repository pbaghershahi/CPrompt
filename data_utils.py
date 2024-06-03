import torch, random, os
import numpy as np
from torch_geometric.datasets import QM9, TUDataset, CitationFull, Planetoid, Airports
from utils import *
from model import *
import pandas as pd
from torch_geometric.loader import DataLoader as PyG_Dataloader
from torch_geometric.data import Data, Batch, Dataset as PyG_Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph, dense_to_sparse
from torch_geometric.loader import NeighborLoader
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.datasets import make_spd_matrix
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
from copy import deepcopy
from sklearn.utils import shuffle as sk_shuffl
import ipdb
import gc


def graph_collate(batch):
    if not isinstance(batch, list):
        g_list = []
        for g in batch:
            g_list.extend(g)
    else:
        g_list = batch
    g_batch = Batch.from_data_list(g_list)
    return g_batch


def add_multivariate_noise(features, mean_shift, cov_scale) -> None:
    n_samples, n_feats = features.size()
    mean = np.ones(n_feats) * mean_shift
    cov_matrix = np.eye(n_feats) * cov_scale
    noise = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    features += torch.as_tensor(noise, dtype=torch.float, device=features.device)
    return features


def graph_ds_add_noise(dataset, mean_shift = None, cov_scale = None):
    x_idxs = dataset.slices["x"]
    for class_id in range(dataset.num_classes):
        if mean_shift == None:
            mean_shift = np.random.uniform(-2, 2)
        else:
            mean_shift = np.random.uniform(-mean_shift, mean_shift)
        cov_scale = 1. if cov_scale is None else cov_scale
        temp_idxs = (dataset.y == class_id).nonzero().T[0]
        temp_idxs = torch.cat((x_idxs[temp_idxs][:, None], x_idxs[temp_idxs+1][:, None]), dim=1)
        class_idxs = []
        for i in range(temp_idxs.size(0)):
            class_idxs.append(torch.arange(temp_idxs[i, 0], temp_idxs[i, 1]))
        class_idxs = torch.cat(class_idxs)
        dataset.x[class_idxs, :] = add_multivariate_noise(dataset.x[class_idxs, :], mean_shift, cov_scale)
    return dataset


class DomianShift():
    def __init__(self, n_domains=2, *args, **kwargs) -> None:
        self.n_domains = n_domains

    @classmethod
    def save_to_file(cls, dataset_name, domain_idx_dict, dir_path=None):
        if dir_path is None:
            dir_path = f"./files/{dataset_name}/domain_indicies"
        os.makedirs(dir_path, exist_ok=True)
        files_path = os.path.join(dir_path, f"domain_idicies.txt")
        with open(files_path, "w") as f_handle:
            for domain_id, idxs in domain_idx_dict.items():
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
    
    
class GraphClassification(Dataset):
    def __init__(self, graph, train_per=0.85, test_per=0.15) -> None:
        super().__init__()
        self._data = graph
        if train_per + test_per != 1.0:
            valid_per = 1 - (train_per + test_per)
        else:
            valid_per = 0.0
        print("Total num nodes: ", graph.x.size(0))
        self.n_nodes = len(graph)
        self.n_feats = graph.x.size(1)
        self.n_classes = graph.num_classes
        self.train_idx = int(self.n_nodes * train_per)
        self.n_train = self.train_idx
        self.n_valid = int(self.n_nodes * valid_per)
        self.test_idx = self.train_idx + self.n_valid
        self.n_test = self.n_nodes - self.test_idx

    def _shuffle(self,):
        self._data[:self.train_idx] = self._data[:self.train_idx].shuffle()
        return self._data

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, indices):
        return self._data[indices], self._data.y[indices]

class NodeClassification(Dataset):
    def __init__(self, graph, num_classes=1, n_hopes=2, train_per=0.85, test_per=0.15):
        self._data = graph
        if train_per + test_per != 1.0:
            valid_per = 1 - (train_per + test_per)
        else:
            valid_per = 0.0
        print("Total num nodes: ", graph.x.size(0))
        loader = NeighborLoader(
            self._data,
            input_nodes=torch.arange(graph.x.size(0)),
            num_neighbors=[-1]*n_hopes,
            batch_size=1,
            replace=False,
            shuffle=False,
        )
        print("Neighborhood Loader Created")
        self.all_edges = []
        self.all_nids = []
        self.r_nodes = []
        for i, batch in enumerate(loader):
            if batch.n_id.size(0) > 1:
                print(f"Batch num: {i}/{len(loader)}", end='\r', flush=True)
                self.r_nodes.append(batch.input_id)
                self.all_nids.append(batch.n_id)
                self.all_edges.append(batch.edge_index)
        self.r_nodes = torch.tensor(self.r_nodes)
        self.n_nodes = len(self.all_nids)
        self.n_feats = graph.x.size(1)
        self.n_classes = num_classes
        self.train_idx = int(self.n_nodes * train_per)
        self.n_train = self.train_idx
        self.n_valid = int(self.n_nodes * valid_per)
        self.test_idx = self.train_idx + self.n_valid
        self.n_test = self.n_nodes - self.test_idx

    def _shuffle(self,):
        perm = torch.randperm(self.train_idx)
        self.all_nids[:self.train_idx] = [self.all_nids[i] for i in perm]
        self.all_edges[:self.train_idx] = [self.all_edges[i] for i in perm]
        self.r_nodes[:self.train_idx] = self.r_nodes[perm]

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = [i for i in range(*indices.indices(len(self)))]
        else:
            raise TypeError("Invalid index type. Must be int, slice, or list.")
        all_graphs = []
        for idx in indices:
            x = self._data.x[self.all_nids[idx]]
            y = self._data.y[self.r_nodes[idx]]
            edges = self.all_edges[idx]
            all_graphs.append(Data(x=x, edge_index=edges, y=y))
        return all_graphs, self._data.y[self.r_nodes[indices]]


class RandomNodeDatset(object):
    def __init__(
            self,
            graph_label=0,
            num_nclass=1,
            n_feats=1,
            cov_scale=2,
            init_gmm=None,
            main_ds=True) -> None:
        super().__init__()
        self.num_nclass = num_nclass
        self.graph_label = graph_label
        if init_gmm is not None:
            self.gmm = init_gmm
        if main_ds:
            self.gen_dists(num_nclass, n_feats, cov_scale)

    def gen_dists(self, num_nclass, n_feats, cov_scale):
        means = np.random.uniform(-1, 1, size=(num_nclass, n_feats))
        covariances = [make_spd_matrix(n_feats)*cov_scale for _ in range(num_nclass)]
        # TODO: We can change the probability of each gussian kernel. Currently we set weights to uniform.
        weights = np.ones(num_nclass) / num_nclass
        self.gmm = GaussianMixture(n_components=num_nclass)
        self.gmm.weights_ = weights
        self.gmm.means_ = means
        self.gmm.covariances_ = covariances
        self.gmm.precisions_ = np.linalg.inv(covariances)

    def gen_xy(self,
               node_feats=None,
               node_labels=None,
               num_nsamples=1,
               nlabel_perm=0.0,
               normalize_mode=None) -> None:
        if (node_feats is None) and (node_labels is None):
            self.num_nsamples = num_nsamples
            self.x, self.y = self.gmm.sample(num_nsamples)
            if nlabel_perm > 0:
                perm_mask = np.random.choice(
                    [True, False], (self.y.shape[0],), p=[nlabel_perm, 1-nlabel_perm]
                    )
                perm = np.random.choice(self.y.shape[0], (perm_mask.sum(),))
                self.y[perm_mask] = self.y[perm]
        else:
            self.x, self.y = node_feats, node_labels
        self.n_feats = self.x.shape[1]
        if normalize_mode is not None:
            _ = self.normalize_feats(normalize_mode)

    def normalize_feats(self, mode, axis=0):
        if mode == "max":
            max_value = self.x.max(axis=axis)
            max_value[max_value==0] = 1.
            self.x = self.x/max_value
        else:
            mean = self.x.mean(axis=axis)
            std = self.x.std(axis=axis)
            std = np.maximum(std, np.ones_like(std)*1e-12)
            self.x = (self.x - mean) / std
        return self.x

    def add_multivariate_noise(self, mean, cov_matrix, inplace=True) -> None:
        n_samples, n_feats = self.x.shape
        noise = np.random.multivariate_normal(mean, cov_matrix, n_samples)
        self.x += noise

    def visualize(self, save_path):
        tsne = TSNE(n_components=2, random_state=42)
        x_tsne = tsne.fit_transform(self.x)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=self.y, cmap=plt.cm.Set1, edgecolor='k')
        plt.title("t-SNE Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(save_path)
        plt.close()
        

class RandomGraphDatset(RandomNodeDatset):
    def __init__(self,
                 num_gclass,
                 num_nclass=1,
                 n_feats=1,
                 cov_scale=2,
                 init_gmm=None,
                 main_ds=True) -> None:
        super().__init__(
            num_nclass=num_nclass,
            n_feats=n_feats,
            cov_scale=cov_scale,
            init_gmm=init_gmm,
            main_ds=main_ds
        )
        self.num_gclass = num_gclass

    def gen_ngclass_probs(self,):
        nclass_probs = [i*1/self.num_nclass for i in range(3)]
        comb_dict = OrderedDict()
        for i in range(self.num_gclass):
            gclass_prob = {k:0.0 for k in range(self.num_nclass)}
            total_prob = 0.0
            j = 0
            while total_prob < 1.0:
                c_value = min(1-total_prob, np.random.choice(nclass_probs))
                gclass_prob[j%self.num_nclass] += c_value
                total_prob += c_value
                j += 1
            comb_dict[i] = gclass_prob
        # Permute class probabilities to avoid lack of samples from a specific class.
        for key, value in comb_dict.items():
            nclass_keys = list(value.keys())
            nclass_values = list(value.values())
            random.shuffle(nclass_keys)
            comb_dict[key] = dict(zip(nclass_keys, nclass_values))
        return comb_dict

    def gen_graph_ds(self,
                     num_nsamples,
                     nlabel_perm,
                     ng_perclass,
                     nn_perclass,
                     nnoise_selec_std=-1,
                     ngclass_probs=None) -> None:
        # Generate sampling probabilities of each class of nodes for every graph class
        nselec_std = nnoise_selec_std if nnoise_selec_std > .0 else 1/(2*self.num_nclass)
        self.ngclass_probs = self.gen_ngclass_probs() if ngclass_probs is None else ngclass_probs
        # Identify the number of graphs per grpah class
        self.all_numg = np.array([
            ng_perclass + int(ng_perclass * np.random.choice(np.arange(-0.5, 0.5+0.25, 0.25)))
            for _ in range(self.num_gclass)
            ])
        _, y_counts = np.unique(self.y, return_counts=True)
        self.all_graphs = []
        self.all_labels = []
        self.all_gsizes = {}
        for i, (key, value) in enumerate(self.ngclass_probs.items()):
            # Sort node class probabilities for the graph class of interest.
            c_probs = list(value.items())
            c_probs.sort(key= lambda t: t[0])
            c_probs = np.array([p[1] for p in c_probs])
            # Compute probability of per class samples based on #sample in each class
            c_probs = c_probs/y_counts
            selec_prob = np.empty((num_nsamples,))
            last_idx = 0
            for j, c in enumerate(y_counts):
                selec_prob[last_idx:last_idx+c] = c_probs[j]
                last_idx += c
            # Generate n graphs with different #nodes
            graph_sizes = np.random.randint(nn_perclass[0], nn_perclass[1], (self.all_numg[i]))
            for k in graph_sizes:
                # Draw k sample from all classes of the GMM based on class probabilities
                s_prob = selec_prob
                if nnoise_selec_std > -1:
                    selec_noise = np.random.normal(loc=.0, scale=nselec_std, size=(num_nsamples,))/num_nsamples
                    s_prob += selec_noise
                    s_prob = np.where(s_prob > .0, s_prob, .0)
                    s_prob /= s_prob.sum()
                sample_idxs = np.random.choice(num_nsamples, k, p=s_prob)
                rgraph = RandomNodeDatset(
                    graph_label=i
                    )
                """ TODO: Currently we are assuming that all nodes have the same label of their
                graph, but we need to consider labeling nodes as well. We can do this as PyG does
                to add a one-hot vector of the label to the feature vector of each node."""
                rgraph.gen_xy(node_feats=self.x[sample_idxs, :], node_labels=self.y[sample_idxs])
                rgraph.edges = add_edges(self.y, 3, 0.2)
                self.all_graphs.append(rgraph)
                self.all_labels.append(i)
                # self.all_graphs.append((sample_idxs, i))
            self.all_gsizes[i] = graph_sizes
        self.num_gsamples = len(self.all_graphs)
        perm = np.random.permutation(self.num_gsamples)
        self.all_graphs = [self.all_graphs[i] for i in perm]
        self.all_labels = np.array(self.all_labels)[perm]

    def init_loaders(self, train_per=0.85, test_per=0.15, batch_size=32) -> None:
        if train_per + test_per != 1.0:
            valid_per = 1 - (train_per + test_per)
        else:
            valid_per = 0.0
        self.train_idx = int(self.num_gsamples * train_per)
        self.n_train = self.train_idx
        self.n_valid = int(self.num_gsamples * valid_per)
        self.test_idx = self.train_idx + self.n_valid
        self.n_test = self.num_gsamples - self.test_idx
        train_graphs, valid_graphs, test_graphs = [], [], []
        for i, graph in enumerate(self.all_graphs):
            if i < self.train_idx:
                train_graphs.append(Data(x=graph.x, edge_index=graph.edges, y=torch.tensor(self.all_labels[i])))
            elif self.train_idx <= i < self.test_idx:
                valid_graphs.append(Data(x=graph.x, edge_index=graph.edges, y=torch.tensor(self.all_labels[i])))
            else:
                test_graphs.append(Data(x=graph.x, edge_index=graph.edges, y=torch.tensor(self.all_labels[i])))
        self.train_loader = PyG_Dataloader(train_graphs, batch_size=batch_size, shuffle=True)
        self.valid_loader = PyG_Dataloader(valid_graphs, batch_size=batch_size, shuffle=False)
        self.test_loader = PyG_Dataloader(test_graphs, batch_size=batch_size, shuffle=False)

    def to_tensor(self, all_graphs=True, normalize_mode=False) -> None:
        self.x = torch.as_tensor(self.x)
        if normalize_mode:
            _ = normalize_(self.x, dim=0, mode=normalize_mode)
        self.y = torch.as_tensor(self.y)
        for graph in self.all_graphs:
            graph.x = torch.as_tensor(graph.x, dtype=torch.float32)
            if normalize_mode:
                _ = normalize_(graph.x, dim=0, mode=normalize_mode)
            graph.edges = torch.as_tensor(graph.edges)
            graph.y = torch.as_tensor(graph.y)

    def visualize(self, save_path) -> None:
        x_graphs = []
        y_graphs = []
        for graph in self.all_graphs:
            x_graphs.append(graph.x.mean(axis=0))
            y_graphs.append(graph.graph_label)
        x_graphs = np.array(x_graphs)
        y_graphs = np.array(y_graphs)
        tsne = TSNE(n_components=2, random_state=42)
        x_tsne = tsne.fit_transform(x_graphs)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_graphs, cmap=plt.cm.Set1, edgecolor='k')
        plt.title("t-SNE Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig(save_path)
        plt.close()


def make_datasets(
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
        ):
    colors = np.array([
        "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) \
        for i in range(num_gclass)])

    s_dataset = RandomGraphDatset(
        num_gclass,
        num_nclass,
        n_feats,
        cov_scale,
        main_ds=True
        )
    s_dataset.gen_xy(
        num_nsamples=num_nsamples,
        nlabel_perm=nlabel_perm,
        normalize_mode=norm_mode
        )
    s_dataset.gen_graph_ds(
        num_nsamples,
        nlabel_perm,
        ng_perclass,
        nn_perclass,
        nnoise_selec_std=graph_selec_noise,
        ngclass_probs=None
        )
    s_dataset.to_tensor()
    s_dataset.init_loaders(train_per=train_per, test_per=test_per)

    t_dataset = RandomGraphDatset(
        num_gclass,
        num_nclass,
        init_gmm=s_dataset.gmm,
        main_ds=False
        )
    t_dataset.gen_xy(
        num_nsamples=num_nsamples,
        nlabel_perm=nlabel_perm,
        normalize_mode=None
        )
    mean = np.zeros(t_dataset.x.shape[1])
    cov_matrix = np.eye(t_dataset.x.shape[1]) * cov_scale
    t_dataset.add_multivariate_noise(mean, cov_matrix)
    t_dataset.normalize_feats(mode=norm_mode)
    t_dataset.gen_graph_ds(
        num_nsamples,
        nlabel_perm,
        ng_perclass,
        nn_perclass,
        nnoise_selec_std=graph_selec_noise,
        ngclass_probs=s_dataset.ngclass_probs
        )
    t_dataset.to_tensor()
    t_dataset.init_loaders(train_per=train_per, test_per=test_per)

    if visualize:
        s_dataset.visualize("/content/CPrompt/tsne_source.png")
        t_dataset.visualize("/content/CPrompt/tsne_destination.png")
    
    return s_dataset, t_dataset


def add_edges(labels, p_intra_edge, p_inter_edge):
    unique_labels = np.unique(labels)
    p_intra_edge = to_prob_dict(unique_labels, p_intra_edge)
    p_inter_edge = to_prob_dict(unique_labels, p_inter_edge)
    edges = []
    for c in unique_labels:
        # Making intra-class edges
        intra_idx = np.nonzero(labels == c)[0]
        num_intra_edges = int(intra_idx.shape[0] * p_intra_edge[c])
        intra_s = np.random.choice(intra_idx, num_intra_edges, replace=True)
        intra_t = np.random.choice(intra_idx, num_intra_edges, replace=True)
        not_loops_mask = (intra_s != intra_t)
        intra_s = intra_s[not_loops_mask]
        intra_t = intra_t[not_loops_mask]
        intra_edges = np.stack((intra_s, intra_t))
        # Making inter-class edges
        inter_idx = np.nonzero(labels != c)[0]
        num_inter_edges = int(inter_idx.shape[0] * p_inter_edge[c])
        inter_s = np.random.choice(intra_idx, num_inter_edges, replace=True)
        inter_t = np.random.choice(inter_idx, num_inter_edges, replace=True)
        inter_edges = np.stack((inter_s, inter_t))
        # Class edges
        c_edges = np.concatenate((intra_edges, inter_edges), axis=1)
        c_edges = np.concatenate((c_edges, c_edges[[1, 0], :]), axis=1)
        edges.append(c_edges)
    edges = np.concatenate(edges, axis=1)
    edges = np.asarray(list(set(map(tuple, edges.T)))).T
    edges = edges[:, edges[0, :].argsort()]
    return edges


def to_prob_dict(labels, p):
    if not isinstance(p, dict):
        p = dict(zip(labels, np.ones(labels.shape[0]) * p))
    return p


def remove_edges(labels, edges, p_intra_edge, p_inter_edge):
    
    def get_drop_mask(mask, p):
        edge_idxs = mask.nonzero()[0]
        num_retained_edges = int(edge_idxs.shape[0] * (1 - p))
        perm = np.random.permutation(edge_idxs.shape[0])[:num_retained_edges]
        retained_idxs = edge_idxs[perm]
        mask[retained_idxs] = False
        return ~mask
        
    unique_labels = np.unique(labels)
    p_intra_edge = to_prob_dict(unique_labels, p_intra_edge)
    p_inter_edge = to_prob_dict(unique_labels, p_inter_edge)
    overal_mask = np.ones(edges.shape[1]).astype(bool)
    for c in unique_labels:
        intra_idx = np.nonzero(labels == c)[0]
        intra_mask = np.isin(edges, intra_idx)
        indicator_sum = intra_mask.sum(axis=0)
        intra_mask = (indicator_sum == 2)
        inter_mask = (indicator_sum == 1)
        intra_mask = get_drop_mask(intra_mask, p_intra_edge[c])
        inter_mask = get_drop_mask(inter_mask, p_inter_edge[c])
        overal_mask = overal_mask * intra_mask * inter_mask
    edges = edges[:, ~overal_mask]
    return edges


def add_structural_noise(data, p_intra = None, p_inter = None):
    labels = data.y.numpy()
    edges = data.edge_index.numpy()
    unique_labels = np.unique(labels)
    n_classes = unique_labels.shape[0]
    if isinstance(p_intra, float) or isinstance(p_inter, float):
        p_intra = {
            unique_labels[i]: np.random.choice(np.array([-p_intra, 0.0, p_intra]), replace=True) 
            for i in range(n_classes)
        }
        p_inter = {
            unique_labels[i]: np.random.choice(np.array([-p_inter, 0.0, p_inter]), replace=True) 
            for i in range(n_classes)
        }
    add_p_intra = {key:value if value > 0.0 else 0.0 for key, value in p_intra.items()}
    add_p_inter = {key:value if value > 0.0 else 0.0 for key, value in p_inter.items()}
    drop_p_intra = {key:np.abs(value) if value < 0.0 else 0.0 for key, value in p_intra.items()}
    drop_p_inter = {key:np.abs(value) if value < 0.0 else 0.0 for key, value in p_inter.items()}
    pruned_edges = remove_edges(labels, edges, drop_p_intra, drop_p_inter)
    new_edges = add_edges(labels, add_p_intra, add_p_inter)
    edges = np.concatenate((pruned_edges, new_edges), axis=1)
    data.edge_index = torch.as_tensor(edges)
    return data


class GDataset(nn.Module):
    def __init__(self,):
        pass

    def init_loaders_(self,):
        "Implement this is children classes"
        pass

    def normalize_feats_(self,):
        "Implement this is children classes"
        return "x"

    def init_ds_idxs_(self, train_idxs, valid_idxs, test_idxs, train_test_split, shuffle, seed):
        if (train_idxs is not None) and (valid_idxs is not None) and (test_idxs is not None):
            self.n_train = train_idxs.size(0)
            self.n_valid = valid_idxs.size(0)
            self.n_test = test_idxs.size(0)
            self.train_idxs = train_idxs
            self.valid_idxs = valid_idxs
            self.test_idxs = test_idxs
        else:
            all_idxs = torch.arange(self.num_gsamples)
            if shuffle: 
                fix_seed(seed)
                perm = torch.randperm(self.num_gsamples)
                all_idxs = all_idxs[perm]
            if train_test_split[0] + train_test_split[1] != 1.0:
                valid_per = 1 - (train_test_split[0] + train_test_split[1])
            else:
                valid_per = 0.0
            self.n_train = int(self.num_gsamples * train_test_split[0])
            self.n_valid = int(self.num_gsamples * valid_per)
            self.n_test = self.num_gsamples - (self.n_train + self.n_valid)
            self.train_idxs = all_idxs[:self.n_train]
            self.valid_idxs = all_idxs[self.n_train:self.n_train + self.n_valid]
            self.test_idxs = all_idxs[self.n_train + self.n_valid:self.n_train + self.n_valid + self.n_test]

    def initialize(self,):
        "Implement this is children classes"
        return "x"


class SimpleDataset(Dataset):
    def __init__(self,
                 graph_list: List,
                 **kwargs) -> None:
        super(SimpleDataset, self).__init__()
        self._data = graph_list

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SubsetRandomSampler(SubsetSampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))


class SubgraphSet(Dataset):
    def __init__(self,
                 graph_data,
                 n_hopes,
                 **kwargs) -> None:
        super(SubgraphSet, self).__init__()
        self._data = graph_data
        self.all_nids, self.all_edges = self.init_induced_graphs(smallest_size=10, largest_size=30)

    def init_induced_graphs(self, smallest_size=10, largest_size=30):
        induced_nodes = []
        induced_edge_idxs = []
        for index in range(self._data.x.size(0)):
            current_label = self._data.y[index].item()

            current_hop = 2
            subset, _, _, _ = k_hop_subgraph(
                node_idx=index, num_hops=current_hop, edge_index = self._data.edge_index,
                num_nodes = self._data.x.size(0), relabel_nodes = True
                )

            while len(subset) < smallest_size and current_hop < 5:
                current_hop += 1
                subset, _, _, _ = k_hop_subgraph(
                    node_idx = index, num_hops=current_hop,
                    edge_index = self._data.edge_index,
                    num_nodes = self._data.x.size(0), relabel_nodes = True
                    )

            if len(subset) < smallest_size:
                need_node_num = smallest_size - len(subset)
                pos_nodes = torch.argwhere(self._data.y == int(current_label))
                candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
                candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.size(0))][:need_node_num]
                subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, self._data.edge_index, num_nodes=self._data.x.size(0), relabel_nodes=True)
            induced_nodes.append(subset)
            induced_edge_idxs.append(sub_edge_index)

        return induced_nodes, induced_edge_idxs

    def __len__(self):
        return self._data.x.size(0)

    def __getitem__(self, idx):
        x = self._data.x[self.all_nids[idx]]
        y = self._data.y[idx]
        edges = self.all_edges[idx]
        return Data(x=x, edge_index=edges, y=y)


class NodeToGraphDataset(GDataset):
    def __init__(self,
                 main_dataset,
                 n_hopes = 2,
                 **kwargs) -> None:
        super(NodeToGraphDataset, self).__init__()
        if isinstance(main_dataset, PyG_Dataset):
            self._data = main_dataset._data
        elif isinstance(main_dataset, Data):
            self._data = main_dataset
        else:
            raise "Data type is not supported!"
        self.n_feats = main_dataset.x.size(1)
        self.num_nsamples = main_dataset.x.size(0)
        self.num_nclass = main_dataset.y.unique().size(0)
        self.num_gclass = self.num_nclass
        self.num_gsamples = self.num_nsamples
        self.n_hopes = n_hopes

    @property
    def x(self,):
        return self._data.x

    def normalize_feats_(self, normalize_mode, **kwargs):
        self.train_ds._data.x, train_normal_params = normalize_(self.train_ds._data.x, dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self.valid_ds._data.x, _ = normalize_(
                self.valid_ds._data.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds._data.x, _ = normalize_(
                self.test_ds._data.x, dim=0, 
                mode=normalize_mode, normal_params = train_normal_params)

    def init_loaders_(self, batch_size, loader_collate = graph_collate):
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)

    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split = [0.85, 0.15],
            loader_collate = graph_collate,
            batch_size = 32, 
            normalize_mode = None,
            shuffle = False, **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )   
        self.train_ds = SubgraphSet(
            deepcopy(self._data.subgraph(self.train_idxs)),
            n_hopes = self.n_hopes
            )
        self.valid_ds = SubgraphSet(
            deepcopy(self._data.subgraph(self.valid_idxs)),
            n_hopes = self.n_hopes
            )
        self.test_ds = SubgraphSet(
            deepcopy(self._data.subgraph(self.test_idxs)),
            n_hopes = self.n_hopes
            )
        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.init_loaders_(batch_size, loader_collate)


def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)


class InfluenceDataSet(Dataset):
    def __init__(self, file_dir):
        adj_matrices = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.int8)
        # self-loop trick, the input adj_matrices should have no self-loop
        # identity = np.identity(adj_matrices.shape[1])
        # adj_matrices += identity
        # adj_matrices[adj_matrices != 0] = 1.0
        # ipdb.set_trace()
        influence_features = np.load(os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        labels = np.load(os.path.join(file_dir, "label.npy"))
        vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        embedding = load_w2v_feature(os.path.join(file_dir, "deepwalk.emb_64"), vertices.max())
        vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
        # vertex_features = preprocessing.scale(vertex_features)
        # if shuffle:
        #     adj_matrices, influence_features, labels, vertices = sk_shuffl(
        #         adj_matrices, influence_features,
        #         labels, vertices,
        #         random_state = 2728
        #         )
        self.vertices = torch.as_tensor(vertices, dtype=torch.int)[:50000]
        self.adj_matrices = torch.as_tensor(adj_matrices, dtype=torch.int8)[:50000]
        self.labels = torch.as_tensor(labels, dtype=torch.int64)[:50000]
        self.vertex_features = torch.as_tensor(vertex_features, dtype=torch.float)
        self.influence_features = torch.as_tensor(influence_features, dtype=torch.float)
        self.embedding = torch.as_tensor(embedding, dtype=torch.float)

    def __len__(self):
        return self.adj_matrices.size(0)

    def __getitem__(self, idx):
        node_ids = self.vertices[idx]
        x = torch.cat([
            self.embedding[node_ids],
            self.vertex_features[node_ids],
            self.influence_features[idx]
            ], dim=-1)
        edges, _ = dense_to_sparse(self.adj_matrices[idx])
        y = self.labels[idx]
        return Data(x=x, edge_index=edges, y=y)


class EgoNetworkDataset(GDataset):
    def __init__(self,
                 ds_path,
                 **kwargs) -> None:
        super(EgoNetworkDataset, self).__init__()
        self._data = InfluenceDataSet(ds_path)
        self.num_nsamples = self._data.vertex_features.size(0)
        self.num_nclass = self._data.labels.unique().size(0)
        self.num_gclass = self.num_nclass
        self.num_gsamples = self._data.adj_matrices.size(0)
        class_weight = self.num_gsamples / (self.num_gclass * torch.bincount(self._data.labels))
        self.class_weight = torch.as_tensor(class_weight, dtype=torch.float)

    @property
    def n_feats(self,):
        return self._data.influence_features.size(-1) + self._data.embedding.size(-1) + self._data.vertex_features.size(-1)

    def normalize_feats_(self, normalize_mode, **kwargs):
        self._data.embedding[self.train_idxs, :], train_normal_params_embed = normalize_(
            self._data.embedding[self.train_idxs, :],
            dim=0, mode=normalize_mode)
        self._data.vertex_features[self.train_idxs, :], train_normal_params_vfeat = normalize_(
            self._data.vertex_features[self.train_idxs, :],
            dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self._data.embedding[self.valid_idxs, :], _ = normalize_(
                self._data.embedding[self.valid_idxs, :],
                dim=0, mode=normalize_mode, normal_params = train_normal_params_embed)
            self._data.vertex_features[self.valid_idxs, :], _ = normalize_(
                self._data.vertex_features[self.valid_idxs, :],
                dim=0, mode=normalize_mode, normal_params = train_normal_params_vfeat)
        if self.n_test > 0:
            self._data.embedding[self.test_idxs, :], _ = normalize_(
                self._data.embedding[self.test_idxs, :],
                dim=0, mode=normalize_mode, normal_params = train_normal_params_embed)
            self._data.vertex_features[self.test_idxs, :], _ = normalize_(
                self._data.vertex_features[self.test_idxs, :],
                dim=0, mode=normalize_mode, normal_params = train_normal_params_vfeat)

    def init_loaders_(self, batch_size, loader_collate = graph_collate):
        self.train_loader = DataLoader(
            self._data,
            batch_size = batch_size,
            sampler = SubsetRandomSampler(self.train_idxs),
            collate_fn = loader_collate,
            num_workers=1
        )
        self.valid_loader = DataLoader(
            self._data,
            batch_size = batch_size,
            sampler = SubsetSampler(self.valid_idxs),
            collate_fn = loader_collate,
            num_workers=1
        )
        self.test_loader = DataLoader(
            self._data,
            batch_size = batch_size,
            sampler = SubsetSampler(self.test_idxs),
            collate_fn = loader_collate,
            num_workers=1
        )   

    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split=[0.85, 0.15],
            loader_collate = graph_collate,
            batch_size=32,
            normalize_mode = None,
            shuffle = False, **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.init_loaders_(batch_size, loader_collate)


class FromPyGGraph(GDataset):
    def __init__(self,
                 main_dataset: PyG_Dataset,
                 **kwargs) -> None:
        super(FromPyGGraph, self).__init__()
        self._data = deepcopy(main_dataset)
        self.n_feats = self._data.x.size(1)
        self.num_nsamples = self._data.x.size(0)
        self.num_nclass = self._data.num_node_labels
        self.num_gclass = self._data.num_classes
        self.num_gsamples = len(self._data)
        class_weight = self.num_gsamples / (self.num_gclass * torch.bincount(self._data.y))
        self.class_weight = torch.as_tensor(class_weight, dtype=torch.float)

    def gen_graph_ds(self, dataset):
        if len(dataset) == 0:
            return []
        x_idxs = dataset.slices["x"]
        all_graphs = []
        for i in range(x_idxs.size(0)-1):
            g = dataset[i]
            temp_g = Data(
                x = dataset._data.x[x_idxs[i]:x_idxs[i+1], :], 
                edge_index = g.edge_index, y = g.y
            )
            all_graphs.append(temp_g)
        return all_graphs

    def normalize_feats_(self, normalize_mode, **kwargs):
        self.train_ds._data.x, train_normal_params = normalize_(self.train_ds._data.x, dim=0, mode=normalize_mode)
        if self.n_valid > 0:
            self.valid_ds._data.x, _ = normalize_(self.valid_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        if self.n_test > 0:
            self.test_ds._data.x, _ = normalize_(self.test_ds._data.x, dim=0, mode=normalize_mode, normal_params = train_normal_params)
        
    def init_loaders_(self, batch_size, loader_collate = graph_collate):
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=loader_collate, num_workers=1)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, collate_fn=loader_collate, num_workers=1)
        
    def initialize(
            self,
            train_idxs: torch.Tensor = None,
            valid_idxs: torch.Tensor = None,
            test_idxs: torch.Tensor = None,
            train_test_split = [0.85, 0.15],
            loader_collate = graph_collate,
            batch_size = 32, 
            normalize_mode = None,
            shuffle = False, **kwargs) -> None:
        self.init_ds_idxs_(
            train_idxs = train_idxs, valid_idxs = valid_idxs, test_idxs = test_idxs,
            train_test_split = train_test_split,
            shuffle = shuffle, seed = kwargs["seed"] if "seed" in kwargs else 2411
        )
        self.train_ds = self._data.copy(self.train_idxs)
        self.valid_ds = self._data.copy(self.valid_idxs) if self.n_valid > 0 else []
        self.test_ds = self._data.copy(self.test_idxs) if self.n_test > 0 else []
        if normalize_mode is not None:
            self.normalize_feats_(normalize_mode)
        self.train_ds = SimpleDataset(self.gen_graph_ds(self.train_ds))
        self.valid_ds = SimpleDataset(self.gen_graph_ds(self.valid_ds))
        self.test_ds = SimpleDataset(self.gen_graph_ds(self.test_ds))
        self.init_loaders_(batch_size, loader_collate)
        

class GenDataset(object):
    def __init__(self, logger) -> None:
        self.logger = logger

    def get_graph_da_dataset(
            self,
            sds_name,
            tds_name,
            store_to_path = "./data",
            train_per = 0.80,
            test_per = 0.20,
            batch_size = 32,
            norm_mode = "max",
            node_attributes = True,
            seed = 2411
            ):

        s_dataset = TUDataset(
            root = store_to_path,
            name = sds_name,
            use_node_attr = node_attributes
        )
        s_dataset = FromPyGGraph(s_dataset)
        s_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            seed = seed
        )

        t_dataset = TUDataset(
            root = store_to_path,
            name = tds_name,
            use_node_attr = node_attributes
        )
        t_dataset = FromPyGGraph(t_dataset)
        t_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            seed = seed
        )
        return s_dataset, t_dataset 


    def get_node_da_dataset(
            self,
            sds_name,
            tds_name,
            store_to_path = "./data",
            train_per = 0.80,
            test_per = 0.20,
            batch_size = 32,
            norm_mode = "max",
            n_hopes = 2,
            node_attributes = True
            ):

        s_dataset = Airports(
            root = store_to_path,
            name = sds_name,
        )
        s_dataset = NodeToGraphDataset(
            s_dataset,
            n_hopes = n_hopes
            )
        s_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = False
            )

        t_dataset = Airports(
            root = store_to_path,
            name = tds_name,
            )
        t_dataset = NodeToGraphDataset(
            t_dataset,
            n_hopes = n_hopes
            )
        t_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = False
            )
        return s_dataset, t_dataset


    def get_node_dataset(
        self,
        ds_name,
        shift_type = "structural",
        p_intra = 0.0,
        p_inter = 0.0,
        cov_scale = 2,
        mean_shift = 0,
        shift_mode = "class_wise",
        train_per = 0.85,
        test_per = 0.15,
        batch_size = 32,
        n_hopes = 2,
        norm_mode = "max",
        node_attributes = True,
        seed = 2411
    ):

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
        other_idxs = torch.nonzero(other_mask).view(-1)
        n_train = train_idxs.size(0)
        n_val = valid_idxs.size(0)
        n_test = test_idxs.size(0)
        n_other = other_idxs.size(0)
        fix_seed(seed)
        train_perm = torch.randperm(n_train)
        valid_perm = torch.randperm(n_val)
        test_perm = torch.randperm(n_test)
        other_perm = torch.randperm(n_other)
        s_perm = torch.cat([
            train_idxs[train_perm[:n_train//2]],
            valid_idxs[valid_perm[:n_val//2]],
            test_idxs[test_perm[:n_test//2]],
            other_idxs[other_perm[:n_other//2]],
            ])
        t_perm = torch.cat([
            train_idxs[train_perm[n_train//2:]],
            valid_idxs[valid_perm[n_val//2:]],
            test_idxs[test_perm[n_test//2:]],
            other_idxs[other_perm[n_other//2:]],
            ])
        s_data = deepcopy(dataset._data.subgraph(s_perm))
        t_data = deepcopy(dataset._data.subgraph(t_perm))
        s_dataset = NodeToGraphDataset(
            s_data,
            n_hopes = n_hopes
            )
        s_n_train = train_perm[:n_train//2].size(0)
        s_n_valid = valid_perm[:n_val//2].size(0)
        s_n_test = test_perm[:n_test//2].size(0)
        s_dataset.initialize(
            train_idxs = torch.arange(s_n_train),
            valid_idxs = torch.arange(s_n_train, s_n_train + s_n_valid),
            test_idxs = torch.arange(s_n_train + s_n_valid, s_n_train + s_n_valid + s_n_test),
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = False
            )
        if shift_type == "feature":
            if shift_mode == "class_wise":
                t_data = graph_ds_add_noise(t_data, mean_shift, cov_scale)
            else:
                t_data.x[torch.arange(t_data.x.size(0)), :] = add_multivariate_noise(
                    t_data.x[torch.arange(t_data.x.size(0)), :], mean_shift, cov_scale
                )
        elif shift_type == "structural":
            t_data = add_structural_noise(t_data, p_intra, p_inter)
        else:
            print("shift type is not implemented yet")
        t_dataset = NodeToGraphDataset(
            t_data,
            n_hopes = n_hopes
            )
        t_n_train = train_perm[n_train//2:].size(0)
        t_n_valid = valid_perm[n_val//2:].size(0)
        t_n_test = test_perm[n_test//2:].size(0)
        t_dataset.initialize(
            train_idxs = torch.arange(t_n_train),
            valid_idxs = torch.arange(t_n_train, t_n_train + t_n_valid),
            test_idxs = torch.arange(t_n_train + t_n_valid, t_n_train + t_n_valid + t_n_test),
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = False
            )
        
        return s_dataset, t_dataset


    def get_graph_dataset(
        self,
        ds_name,
        shift_type = "structural",
        p_intra = 0.0,
        p_inter = 0.0,
        cov_scale = 2,
        mean_shift = 0,
        shift_mode = "class_wise",
        store_to_path = "./data",
        train_per = 0.80,
        test_per = 0.20,
        batch_size = 32,
        norm_mode = "max",
        node_attributes = True,
        seed = 2411,
    ):
        
        """ Currently supported datasets: 
            - ENZYMES
            - PROTEINS_full
        """
        dataset = TUDataset(
            root = store_to_path,
            name = ds_name,
            use_node_attr = node_attributes
        )

        ntotal_graphs = len(dataset)
        fix_seed(seed)
        perm = torch.randperm(ntotal_graphs)
        s_perm = perm[:int(ntotal_graphs*0.5)]
        t_perm = perm[int(ntotal_graphs*0.5):]
        domain_idx_dict = {0:s_perm, 1:t_perm}
        DomianShift.save_to_file(ds_name, domain_idx_dict)
        s_ds = dataset.copy(s_perm)
        t_ds = dataset.copy(t_perm)

        s_dataset = FromPyGGraph(s_ds)
        s_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            seed = seed
        )

        if shift_type == "feature":
            if shift_mode == "class_wise":
                t_ds = graph_ds_add_noise(t_ds, mean_shift, cov_scale)
            else:
                t_ds.x[torch.arange(t_ds.x.size(0)), :] = add_multivariate_noise(
                    t_ds.x[torch.arange(t_ds.x.size(0)), :], mean_shift, cov_scale
                )
        elif shift_type == "structural":
            t_ds = add_structural_noise(t_ds, p_intra, p_inter)
        else:
            print("shift type is not implemented yet")
        t_dataset = FromPyGGraph(t_ds)
        t_dataset.initialize(
            train_test_split = [train_per, test_per],
            batch_size = batch_size,
            normalize_mode = norm_mode,
            shuffle = True,
            seed = seed
        )
        return s_dataset, t_dataset


    def get_gda_dataset(
            self,
            ds_dir,
            s_ds_name,
            t_ds_name,
            train_per = 0.85,
            test_per = 0.15,
            batch_size = 32,
            get_s_dataset = True,
            get_t_dataset = True,
            seed = 2411
            ):
        if get_s_dataset:
            s_path = ds_dir + s_ds_name
            s_dataset = EgoNetworkDataset(s_path)
            s_dataset.initialize(
                train_test_split = [train_per, test_per],
                batch_size = batch_size,
                shuffle = True,
                seed = seed
            )
        else:
            s_dataset = "s_dataset"
        # gc.collect()
        if get_t_dataset:
            t_path = ds_dir + t_ds_name
            t_dataset = EgoNetworkDataset(t_path)
            t_dataset.initialize(
                train_test_split = [train_per, test_per],
                batch_size = batch_size,
                shuffle = True,
                seed = seed
            )
        else:
            t_dataset = "t_dataset"
        return s_dataset, t_dataset