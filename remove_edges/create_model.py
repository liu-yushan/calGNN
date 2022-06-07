from torch.optim import Adam

from models import GCN, GAT, GAT_GCNParameters


def create_gcn_model(dataset, nhid=16, dropout=0., lr=0.01, weight_decay=5e-4):
    model = GCN(dataset, nhid, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


def create_gat_model(dataset, nhid=8, first_heads=8, dropout=0.6):
    if dataset.name in ["Cora", "Citeseer"]:
        output_heads = 1
        lr = 0.005
        weight_decay = 5e-4
    elif dataset.name == "Pubmed":
        output_heads = 8
        lr = 0.01
        weight_decay = 0.001
    model = GAT(dataset, nhid, first_heads, output_heads, dropout)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer


def create_gat_gcnparam_model(dataset):
    lr = 0.01
    weight_decay = 5e-4
    model = GAT_GCNParameters(dataset)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer
