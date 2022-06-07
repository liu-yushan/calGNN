import argparse


datasets = ["Cora", "Citeseer", "Pubmed"]
models = ["gcn", "gat"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=datasets)
    parser.add_argument("--model", type=str, choices=models)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stopping", type=bool, default=False)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--add_cal_loss", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--lmbda", type=float, default=1.0)
    args = parser.parse_args()
    return args
