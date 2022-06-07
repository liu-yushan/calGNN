# Implementation of GCN and GAT based on
# https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/citation

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(dataset.num_features, nhid)
        self.gc2 = GCNConv(nhid, dataset.num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data, edge_index):
        x = data.x
        a = edge_index.to(x.device)
        x = self.gc1(x, a)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, a)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, dataset, nhid, first_heads, output_heads, dropout):
        super(GAT, self).__init__()
        self.gc1 = GATConv(
            dataset.num_features, nhid, heads=first_heads, dropout=dropout
        )
        self.gc2 = GATConv(
            nhid * first_heads,
            dataset.num_classes,
            heads=output_heads,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data, edge_index):
        x = data.x
        a = edge_index.to(x.device)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, a)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, a)
        return F.log_softmax(x, dim=1)


class GAT_GCNParameters(nn.Module):
    def __init__(self, dataset, nhid=8, first_heads=8, output_heads=1, dropout=0.):
        super(GAT_GCNParameters, self).__init__()
        self.gc1 = GATConv(dataset.num_features, nhid, heads=first_heads)
        self.gc2 = GATConv(
            nhid * first_heads, dataset.num_classes, heads=output_heads, concat=False,
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data, edge_index):
        x = data.x
        a = edge_index.to(x.device)
        x = self.gc1(x, a)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, a)
        return F.log_softmax(x, dim=1)
