import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from utils import *
from torch_geometric.loader import NeighborLoader
from torch.utils.data import Dataset
from sklearn.datasets import make_spd_matrix
from sklearn.mixture import GaussianMixture
from collections import OrderedDict

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.head = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x_adj_list, decoder=True):
        if not decoder:
            scores = scores = self.head(g_embeds)
            return scores, ""
        g_embeds = []
        for i, (x, adj) in enumerate(x_adj_list):
            x = self.gc1(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            g_embeds.append(x.mean(dim=0))
        g_embeds = torch.stack(g_embeds)
        scores = self.head(g_embeds)
        return scores, g_embeds


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_x, adj):
        support = torch.mm(input_x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class LinkPredictionPrompt(nn.Module):
    def __init__(self,
                 emb_dim,
                 h_dim,
                 output_dim,
                 prompt_fn = "trans_x",
                 token_num = 30,
                 device="cuda:0") -> None:
        super(LinkPredictionPrompt, self).__init__()
        self.dropout = 0.2
        self.device = torch.device(device)
        self.head = nn.Linear(h_dim, output_dim)
        if prompt_fn == "trans_x":
            self.linear1 = nn.Linear(emb_dim, h_dim)
            self.prompt = self.trans_x
        elif prompt_fn == "gnn":
            self.gnn1 = GCNConv(emb_dim, h_dim)
            self.bn1 = nn.BatchNorm1d(h_dim)
            self.gnn2 = GCNConv(h_dim, h_dim)
            self.bn2 = nn.BatchNorm1d(h_dim)
            self.prompt = self.gnn
        elif prompt_fn == "add_tokens":
            cross_prune=0.1
            inner_prune=0.3
            self.inner_prune = inner_prune
            self.cross_prune = cross_prune
            self.token_embeds = torch.nn.Parameter(torch.empty(token_num, emb_dim))
            torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            self.pg = self.pg_construct()
            self.prompt = self.add_token

    def pg_construct(self,):
        token_sim = torch.mm(self.token_embeds, torch.transpose(self.token_embeds, 0, 1))
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().t().contiguous()
        pg = Data(x=self.token_embeds, edge_index=edge_index, y=torch.tensor([0]).long())
        return pg

    def simmatToadj(self, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.triu(diagonal=1)
        adj_list = torch.where(adjacency_matrix >= 0.5)
        edge_weights = adjacency_matrix[adj_list]
        adj_list = torch.cat(
            (adj_list[0][None, :], adj_list[1][None, :]),
            dim=0)
        return adj_list, edge_weights
    
    def add_token(self, graphs):
        # pg = self.inner_structure_update()
        self.pg.to(graphs[0].x.device)
        inner_edge_index = self.pg.edge_index
        token_num = self.pg.x.shape[0]
        for graph in graphs:
            g_edge_index = graph.edge_index + token_num
            cross_dot = torch.mm(self.pg.x, torch.transpose(graph.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            x = torch.cat([self.pg.x, graph.x], dim=0)
            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            graph.edge_index = edge_index
            graph.x = x
        return graphs

    def gnn(self, graphs):
        for graph in graphs:
            emb_out = graph.x.squeeze()
            emb_out = self.gnn1(emb_out, graph.edge_index)
            emb_out = F.relu(emb_out)
            emb_out = F.dropout(emb_out, self.dropout, training=self.training)
            emb_out = self.gnn2(emb_out, graph.edge_index)
            emb_out = F.relu(emb_out)
            emb_out = F.dropout(emb_out, self.dropout, training=self.training)
            emb_out = self.head(emb_out)
            graph.x = emb_out
        return graphs

    def trans_x(self, graphs):
        for graph in graphs:
            emb_out = graph.x.squeeze()
            emb_out = F.relu(self.linear1(emb_out))
            emb_out = F.dropout(emb_out, self.dropout, training=self.training)
            emb_out = self.head(emb_out)
            graph.x = emb_out
        return graphs

    def forward(self, graphs):
        return self.prompt(graphs)
    

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

    def gen_edges(self, p_intra_edge, p_inter_edge) -> None:
        self.edges = []
        for c in np.unique(self.y):
            # Making intra-class edges
            intra_idx = np.nonzero(self.y==c)[0]
            num_intra_edges = intra_idx.shape[0]*p_intra_edge
            intra_s = np.random.choice(intra_idx, num_intra_edges, replace=True)
            intra_t = np.random.choice(intra_idx, num_intra_edges, replace=True)
            not_loops_mask = (intra_s != intra_t)
            intra_s = intra_s[not_loops_mask]
            intra_t = intra_t[not_loops_mask]
            intra_edges = np.stack((intra_s, intra_t))
            # Making inter-class edges
            inter_idx = np.nonzero(self.y!=c)[0]
            num_inter_edges = int(intra_idx.shape[0]*p_inter_edge)
            inter_s = np.random.choice(intra_idx, num_inter_edges, replace=True)
            inter_t = np.random.choice(inter_idx, num_inter_edges, replace=True)
            inter_edges = np.stack((inter_s, inter_t))
            # Class edges
            c_edges = np.concatenate((intra_edges, inter_edges), axis=1)
            c_edges = np.concatenate((c_edges, c_edges[[1, 0], :]), axis=1)
            self.edges.append(c_edges)
        self.edges = np.concatenate(self.edges, axis=1)
        self.edges = np.asarray(list(set(map(tuple, self.edges.T)))).T
        self.edges = self.edges[:, self.edges[0, :].argsort()]

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
        self.all_graphs = []
        self.all_labels = []
        self.all_gsizes = {}
        for i, (key, value) in enumerate(self.ngclass_probs.items()):
            # Sort node class probabilities for the graph class of interest.
            c_probs = list(value.items())
            c_probs.sort(key= lambda t: t[0])
            c_probs = np.array([p[1] for p in c_probs])
            # Compute probability of per class samples based on #sample in each class
            _, y_counts = np.unique(self.y, return_counts=True)
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
                rgraph.gen_edges(3, 0.2)
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
        self.train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

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

class GDataset(nn.Module):
    def __init__(self,):
        pass

    def init_loaders(self,):
        "Implement this is children classes"
        pass

    def normalize_feats(self,):
        "Implement this is children classes"
        return "x"

    def visualize(self,):
        "Implement this is children classes"
        pass

class ToGraphDataset(GDataset):
    def __init__(self,
                 main_dataset,
                 normalize=False,
                 **kwargs) -> None:
        super(ToGraphDataset, self).__init__()
        self.n_feats = main_dataset.x.size(1)
        self.num_nsamples = main_dataset.x.size(0)
        self.num_nclass = main_dataset.num_node_labels
        self.num_gclass = main_dataset.num_classes
        self.num_gsamples = len(main_dataset)
        self.x = deepcopy(main_dataset.x)
        if normalize:
            self.normalize_feats(kwargs["normalize_mode"])

    def gen_graph_ds(self, dataset):
        x_idxs = dataset.slices["x"]
        self.all_graphs = []
        for i in range(x_idxs.size(0)-1):
            g = dataset[i]
            temp_g = Data(x=self.x[x_idxs[i]:x_idxs[i+1], :], edge_index=g.edge_index, y=g.y)
            self.all_graphs.append(temp_g)

    def init_loaders(self, train_per=0.85, test_per=0.15, batch_size=32, shuffle=True) -> None:
        if shuffle:
            perm = list(range(len(self.all_graphs)))
            random.shuffle(perm)
            self.all_graphs = [self.all_graphs[idx] for idx in perm]
        if train_per + test_per != 1.0:
            valid_per = 1 - (train_per + test_per)
        else:
            valid_per = 0.0
        self.train_idx = int(self.num_gsamples * train_per)
        self.n_train = self.train_idx
        self.n_valid = int(self.num_gsamples * valid_per)
        self.test_idx = self.train_idx + self.n_valid
        self.n_test = self.num_gsamples - self.test_idx
        self.train_loader = DataLoader(self.all_graphs[:self.n_train], batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.all_graphs[self.n_train:self.test_idx], batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.all_graphs[self.test_idx:], batch_size=batch_size, shuffle=False)

    def visualize(self, save_path) -> None:
        x_graphs = []
        y_graphs = []
        for graph in self.all_graphs:
            x_graphs.append(graph.x.mean(axis=0))
            y_graphs.append(graph.y)
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

    def add_multivariate_noise(self, mean, cov_matrix, inplace=True) -> None:
        n_samples, n_feats = self.x.shape
        noise = np.random.multivariate_normal(mean, cov_matrix, n_samples)
        self.x += torch.as_tensor(noise, dtype=torch.float, device=self.x.device)

    def normalize_feats(self, normalize_mode="max"):
        self.x = normalize_(self.x, dim=0, mode=normalize_mode)

class HeavyPrompt(nn.Module):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.3, trans_x=False):
        super(HeavyPrompt, self).__init__()
        self.inner_prune = inner_prune
        self.cross_prune = cross_prune
        self.token_embeds = torch.nn.Parameter(torch.empty(token_num, token_dim))
        torch.nn.init.kaiming_uniform_(self.token_embeds, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        self.linear1 = nn.Linear(token_dim, token_dim*2)
        self.linear2 = nn.Linear(token_dim*2, token_dim)
        self.trans_x = trans_x
        self.pg = self.pg_construct()

    def pg_construct(self,):
        token_sim = torch.mm(self.token_embeds, torch.transpose(self.token_embeds, 0, 1))
        token_sim = torch.sigmoid(token_sim)
        inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
        edge_index = inner_adj.nonzero().t().contiguous()
        pg = Data(x=self.token_embeds, edge_index=edge_index, y=torch.tensor([0]).long())
        return pg

    def forward(self, graph_batch):
        # pg = self.inner_structure_update()
        self.pg.to(graph_batch[0].x.device)
        inner_edge_index = self.pg.edge_index
        token_num = self.pg.x.shape[0]

        re_graph_list = []
        for g in graph_batch:
            if self.trans_x:
                x = F.relu(self.linear1(g.x))
                x = F.dropout(x, 0.2, training=self.training)
                x = self.linear2(x)
                edge_index = g.edge_index
            else:
                g_edge_index = g.edge_index + token_num
                cross_dot = torch.mm(self.pg.x, torch.transpose(g.x, 0, 1))
                cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
                cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
                cross_edge_index = cross_adj.nonzero().t().contiguous()
                cross_edge_index[1] = cross_edge_index[1] + token_num
                x = torch.cat([self.pg.x, g.x], dim=0)
                edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            y = g.y
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)
        return re_graph_list

    """ Peyman: this function consideres the case where we have a task header or decoder.
    so in this case they don't multiply the representation matrix of the prompt graph
    by the representation of the nodes from the original graph generated by the gnn."""
    def Tune(self, train_loader, gnn, answering, lossfn, opi, device):
        running_loss = 0.
        for batch_id, train_batch in enumerate(train_loader):
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = self.forward(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, train_batch.y)

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

        return running_loss / len(train_loader)

    """ Peyman: this function consideres the case where we don't have a task header or decoder.
    so in this case they multiply the representation matrix of the prompt graph
    by the representation of the nodes from the original graph generated by the gnn.
    This is like adding a task header or decoder. """
    def TuneWithoutAnswering(self, train_loader, gnn, answering, lossfn, opi, device):
        total_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            emb0 = gnn(batch.x, batch.edge_index, batch.batch)
            pg_batch = self.inner_structure_update()
            pg_batch = pg_batch.to(self.device)
            pg_emb = gnn(pg_batch.x, pg_batch.edge_index, pg_batch.batch)
            # cross link between prompt and input graphs
            dot = torch.mm(emb0, torch.transpose(pg_emb, 0, 1))
            sim = torch.softmax(dot, dim=1)
            loss = lossfn(sim, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)