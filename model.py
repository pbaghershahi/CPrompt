import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from utils import *

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
            scores = scores = self.head(x_adj_list)
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