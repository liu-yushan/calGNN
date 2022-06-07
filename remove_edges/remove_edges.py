import os
import numpy as np
from tqdm import tqdm
import torch

from args import get_args
from get_dataset import get_planetoid_dataset
from create_model import create_gcn_model, create_gat_model, create_gat_gcnparam_model
from utils import reproducibility_seed, training, cal_eval_model


def remove_edges(edge_index, data, remove_rate):
    """
    Remove the egdes in edge_index from the graph.
    :return new edge_index, average node degree
    """
    remove_list = torch.randperm(data.num_edges)[: int(data.num_edges * remove_rate)]
    remove_list, _ = torch.sort(remove_list, descending=True)

    def del_tensor_ele(arr, index):
        arr1 = arr[:, 0:index]
        arr2 = arr[:, index + 1 :]
        return torch.cat((arr1, arr2), dim=1)

    new_edge_index = edge_index
    for col in remove_list:
        new_edge_index = del_tensor_ele(new_edge_index, col)
    # Delete single directed edge
    if len(new_edge_index[0]):
        highest_idx = new_edge_index.max()
    else:
        highest_idx = 0
    for node_i in range(highest_idx + 1):
        nodes_connected_i_top = (
            new_edge_index[0, new_edge_index[1, :] == node_i].detach().cpu()
        )
        nodes_connected_i_down = (
            new_edge_index[1, new_edge_index[0, :] == node_i].detach().cpu()
        )
        top_del_i_list = list(
            set(nodes_connected_i_top.numpy()).difference(
                list(nodes_connected_i_down.numpy())
            )
        )
        down_del_i_list = list(
            set(nodes_connected_i_down.numpy()).difference(
                list(nodes_connected_i_top.numpy())
            )
        )
        # Delte these two lists
        for top_del_i in top_del_i_list:
            col_del_i = list(
                set(
                    torch.where((new_edge_index[1, :] == node_i))[0]
                    .detach()
                    .cpu()
                    .numpy()
                ).intersection(
                    set(
                        torch.where(new_edge_index[0, :] == top_del_i)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )[0]
            new_edge_index = del_tensor_ele(new_edge_index, col_del_i)
        for down_del_i in down_del_i_list:
            col_del_i = list(
                set(
                    torch.where((new_edge_index[0, :] == node_i))[0]
                    .detach()
                    .cpu()
                    .numpy()
                ).intersection(
                    set(
                        torch.where(new_edge_index[1, :] == down_del_i)[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                )
            )[0]
            new_edge_index = del_tensor_ele(new_edge_index, col_del_i)
    return new_edge_index, new_edge_index.shape[1] // 2 / data.num_nodes


def run(edge_index, gnn, optimizer, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res_uncal = []
    for num_run in range(args.num_runs):
        test_acc, _, gnn = training(
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
            edge_index,
        )
        gnn.to(device)

        # Uncal
        ece, marg_ece, nll, test_acc = cal_eval_model(
            gnn, dataset, device, args.dataset, args.model, edge_index
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

    return test_acc0, ece0, marg_ece0


if __name__ == "__main__":
    reproducibility_seed(0, 0)
    args = get_args()
    print(args)
    dataset = get_planetoid_dataset(args.dataset)
    data = dataset[0]

    if not os.path.exists("output/"):
        os.makedirs("output/")

    if args.model == "gcn":
        gnn, optimizer = create_gcn_model(dataset)
    elif args.model == "gat":
        gnn, optimizer = create_gat_model(dataset)
    elif args.model == "gat_gcnparam":
        gnn, optimizer = create_gat_gcnparam_model(dataset)

    acc_list, ece_list, marg_ece_list, avg_node_degree_list = [], [], [], []
    for rate in tqdm(np.arange(1, -0.01, -0.05)):
        edge_index, avg_node_degree = remove_edges(data.edge_index, data, rate)
        test_acc0, ece0, marg_ece0 = run(edge_index, gnn, optimizer, args)

        acc_list.append(test_acc0)
        ece_list.append(ece0)
        marg_ece_list.append(marg_ece0)
        avg_node_degree_list.append(avg_node_degree)

    with open("output/{}_{}_remove.txt".format(args.model, args.dataset), "a") as f:
        f.write("Accuracy")
        f.write("\n")
        f.write(str(acc_list))
        f.write("\n")
        f.write("ECE")
        f.write("\n")
        f.write(str(ece_list))
        f.write("\n")
        f.write("Marg. ECE")
        f.write("\n")
        f.write(str(marg_ece_list))
        f.write("\n")
        f.write("Average node degree")
        f.write("\n")
        f.write(str(avg_node_degree_list))
        f.write("\n")
