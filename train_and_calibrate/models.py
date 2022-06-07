# Implementation of GCN, GAT, SGC, and APPNPNet based on
# https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/citation

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP


class GCN(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(dataset.num_features, nhid)
        self.gc2 = GCNConv(nhid, dataset.num_classes)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    def __init__(self, dataset, K):
        super(SGC, self).__init__()
        self.gc1 = SGConv(dataset.num_features, dataset.num_classes, K=K, cached=True)

    def reset_parameters(self):
        self.gc1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GFNN(nn.Module):
    def __init__(self, dataset, nhid, K):
        super(GFNN, self).__init__()
        self.gc1 = SGConv(dataset.num_features, nhid, K=K, cached=True)
        self.fc1 = nn.Linear(nhid, dataset.num_classes)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class APPNPNet(nn.Module):
    def __init__(self, dataset, nhid, dropout):
        super(APPNPNet, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, nhid)
        self.lin2 = nn.Linear(nhid, dataset.num_classes)
        self.prop1 = APPNP(10, 0.1)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)
