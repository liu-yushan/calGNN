import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNStack(nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim):
        super(GCNStack, self).__init__()
        self.num_layers = num_layers
        self.dropout = 0.5
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(dataset.num_features, hidden_dim))
        for i in range(num_layers - 2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.convs.append(self.build_conv_model(hidden_dim, dataset.num_classes))

    def build_conv_model(self, input_dim, hidden_dim):
        return GCNConv(input_dim, hidden_dim, cached=True)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(1, self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.convs[self.num_layers - 1](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class GATStack(nn.Module):
    def __init__(self, dataset, num_layers, hidden_dim):
        super(GATStack, self).__init__()
        self.num_layers = num_layers
        self.dropout = 0.6
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_dim))
        for i in range(num_layers - 2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.convs.append(self.build_conv_model(hidden_dim, dataset.num_classes))

    def build_conv_model(self, input_dim, hidden_dim):
        return GATConv(input_dim, hidden_dim, heads=1)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(1, self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.convs[self.num_layers - 1](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)


class GCN_width(nn.Module):
    def __init__(self, dataset, nhid=16):
        super(GCN_width, self).__init__()
        self.gc1 = GCNConv(dataset.num_features, nhid)
        self.gc2 = GCNConv(nhid, dataset.num_classes)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_width(nn.Module):
    def __init__(self, dataset, output_heads, nhid, first_heads=8):
        super(GAT_width, self).__init__()
        self.gc1 = GATConv(dataset.num_features, nhid, heads=first_heads)
        self.gc2 = GATConv(
            nhid * first_heads, dataset.num_classes, heads=output_heads, concat=False
        )

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
