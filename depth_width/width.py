import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam

from args import get_args
from get_dataset import get_planetoid_dataset
from utils import reproducibility_seed, training, cal_eval_model
from models import GCN_width, GAT_width


def run(width_layer, dataset, args):
    if args.model == "gcn":
        gnn = GCN_width(dataset, nhid=width_layer)
        lr = 0.01
        wd = 5e-4
    elif args.model == "gat":
        if dataset.name in ["Cora", "Citeseer"]:
            output_heads = 1
            lr = 0.005
            wd = 5e-4
        elif dataset.name == "Pubmed":
            output_heads = 8
            lr = 0.01
            wd = 0.001
        gnn = GAT_width(dataset, output_heads, nhid=width_layer)
    optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=wd)

    res_uncal = []
    for num_run in range(args.num_runs):
        test_acc, _ = training(
            dataset,
            gnn,
            optimizer,
            args.model,
            args.dataset,
            args.epochs,
            args.add_cal_loss,
            args.early_stopping,
            args.patience,
            args.alpha,
            args.lmbda,
            num_run,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gnn.to(device)
        ece, marg_ece, nll, test_acc = cal_eval_model(
            gnn, dataset, device, args.dataset, args.model
        )
        res_uncal.append([ece, marg_ece, nll, test_acc])

    # 0: mean, 1: std
    ece0, marg_ece0, nll0, test_acc0 = np.mean(res_uncal, axis=0) * 100
    nll0 = nll0 / 100
    ece1, marg_ece1, nll1, test_acc1 = (
        np.sqrt(np.var(res_uncal, axis=0)[0]) * 100,
        np.sqrt(np.var(res_uncal, axis=0)[1]) * 100,
        np.sqrt(np.var(res_uncal, axis=0)[2]),
        np.sqrt(np.var(res_uncal, axis=0)[3]) * 100,
    )

    return test_acc0, test_acc1, nll0, nll1, ece0, ece1, marg_ece0, marg_ece1


if __name__ == "__main__":
    reproducibility_seed(0, 0)
    args = get_args()
    dataset = get_planetoid_dataset(args.dataset)
    if not os.path.exists("output/"):
        os.makedirs("output/")
    if not os.path.exists("output/width/"):
        os.makedirs("output/width/")
    path = "output/width/width_{}_{}.txt".format(args.model, args.dataset)

    test_acc_mean_list, nll_mean_list, ece_mean_list, marg_ece_mean_list = (
        [],
        [],
        [],
        [],
    )

    width_range = [8, 16, 32, 64, 128, 256, 512, 1024]
    for width_layer in tqdm(width_range):
        test_acc0, test_acc1, nll0, nll1, ece0, ece1, marg_ece0, marg_ece1 = run(
            width_layer, dataset, args
        )
        test_acc_mean_list.append(test_acc0)
        nll_mean_list.append(nll0)
        ece_mean_list.append(ece0)
        marg_ece_mean_list.append(marg_ece0)

        with open(path, "a") as f:
            f.write("Width: {}".format(width_layer))
            f.write("\n")
            f.write("ECE is: {}\u00B1{}".format(np.round(ece0, 2), np.round(ece1, 2)))
            f.write("\n")
            f.write(
                "Marg. ECE is: {}\u00B1{}".format(
                    np.round(marg_ece0, 2), np.round(marg_ece1, 2)
                )
            )
            f.write("\n")
            f.write(
                "Test accuracy is: {}\u00B1{}".format(
                    np.round(test_acc0, 2), np.round(test_acc1, 2)
                )
            )
            f.write("\n")
            f.write("\n")

    def draw(test_acc_mean_list, ece_mean_list, marg_ece_mean_list, args):
        x_axix = width_range

        fig2, ax2 = plt.subplots()
        ax2.plot(
            x_axix,
            ece_mean_list,
            color="cornflowerblue",
            label="ECE",
            marker="o",
            markersize=4,
        )
        ax2.plot(
            x_axix,
            marg_ece_mean_list,
            color="sandybrown",
            marker="o",
            label="MECE",
            markersize=4,
        )
        ax2.set_xlabel("Hidden dimensions per layer", fontsize=14)
        ax2.set_title("{} on {}".format(args.model.upper(), args.dataset), fontsize=14)
        plt.legend(loc=1, prop={"size": 14})
        ax2.xaxis.set_tick_params(labelsize=13)
        ax2.yaxis.set_tick_params(labelsize=13)
        plt.show()
        fig2.savefig(
            "./output/width/width_{}_{}_ece.png".format(args.model, args.dataset)
        )

        fig1, ax1 = plt.subplots()
        ax1.plot(
            x_axix,
            test_acc_mean_list,
            color="green",
            marker="o",
            label="Acc.",
            markersize=4,
        )
        ax1.set_xlabel("Hidden dimensions per layer", fontsize=14)
        ax1.set_title("{} on {}".format(args.model.upper(), args.dataset), fontsize=14)
        plt.ylim((60, 90))
        ax1.xaxis.set_tick_params(labelsize=13)
        ax1.yaxis.set_tick_params(labelsize=13)
        plt.legend(loc=1, prop={"size": 14})
        plt.show()
        fig1.savefig(
            "./output/width/width_{}_{}_acc.png".format(args.model, args.dataset)
        )

    draw(test_acc_mean_list, ece_mean_list, marg_ece_mean_list, args)
