import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from metrics import lower_bound_scaling_ce


warnings.filterwarnings("ignore")


def reproducibility_seed(a, b):
    torch_init_seed = a
    torch.manual_seed(torch_init_seed)
    numpy_init_seed = b
    np.random.seed(numpy_init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(torch_init_seed)


def training(
    dataset,
    gnn,
    optimizer,
    gnn_name,
    data_name,
    epochs,
    add_cal_loss,
    early_stopping,
    patience,
    alpha,
    lmbda,
    num_run,
    save_model=True,
):
    data = dataset[0]
    avg_conf_list, avg_acc_list = [], []

    def training_step(data, alpha, lmbda, epoch, epochs, add_cal_loss, device):
        gnn.train()
        optimizer.zero_grad()
        logits = gnn(data)[data.train_mask]
        nll_loss = F.nll_loss(logits, data.y[data.train_mask])
        if add_cal_loss:
            loss_cal = cal_loss(
                data.y[data.train_mask], logits, lmbda, epoch, epochs, device
            )
            loss = alpha * nll_loss + (1.0 - alpha) * loss_cal
        else:
            loss = nll_loss
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_step(data):
        gnn.eval()
        (
            logits,
            logits_list,
            probs_list,
            accs_list,
            losses_list,
            y_pred_list,
            y_true_list,
        ) = (gnn(data), [], [], [], [], [], [])
        probs_pred = F.softmax(logits, dim=1)

        for _, mask in data("train_mask", "val_mask", "test_mask"):
            y_pred = logits[mask].max(1)[1]
            loss = F.nll_loss(logits[mask], data.y[mask])
            acc = y_pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            logits_list.append(logits[mask].detach().cpu().numpy())
            probs_list.append(probs_pred[mask].detach().cpu().numpy())
            accs_list.append(acc)
            losses_list.append(loss.item())
            y_pred_list.append(y_pred.detach().cpu().numpy())
            y_true_list.append(data.y[mask].detach().cpu().numpy())

        return logits_list, probs_list, accs_list, losses_list, y_pred_list, y_true_list

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnn, data = gnn.to(device), data.to(device)
    gnn.reset_parameters()

    checkpoints_path = "checkpoints/{}_{}/".format(gnn_name, data_name)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    best_val_acc, best_test_acc = 0, 0
    best_val_loss = float("inf")
    val_loss_history = []
    patience_cur = patience

    for epoch in range(epochs):
        training_step(data, alpha, lmbda, epoch, epochs, add_cal_loss, device)
        (
            [logits_train, logits_val, logits_test],
            [probs_train, probs_val, probs_test],
            [train_acc, val_acc, test_acc],
            [train_loss, val_loss, test_loss],
            [y_pred_train, y_pred_val, y_pred_test],
            [y_true_train, y_true_val, y_true_test],
        ) = test_step(data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_cur = patience
            if save_model:
                torch.save(gnn.state_dict(), checkpoints_path + str(num_run))
        else:
            patience_cur -= 1
            if early_stopping and not patience_cur:
                break

    if not early_stopping:
        torch.save(gnn.state_dict(), checkpoints_path + str(num_run))

    return best_test_acc, best_val_acc


def cal_loss(y_true, logits, lmbda, epoch, epochs, device):
    def calculate_confidence_vec(confidence, y_pred, y_true, device, bin_num=15):
        def compute_binned_acc_conf(
            conf_thresh_lower, conf_thresh_upper, conf, pred, true, device
        ):
            filtered_tuples = [
                x
                for x in zip(pred, true, conf)
                if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
            ]
            if len(filtered_tuples) < 1:
                return (
                    torch.tensor(0.0).to(device),
                    torch.tensor(0.0, requires_grad=True).to(device),
                    torch.tensor(0).to(device),
                )
            else:
                correct = len(
                    [x for x in filtered_tuples if x[0] == x[1]]
                )  # How many correct labels
                len_bin = torch.tensor(len(filtered_tuples)).to(
                    device
                )  # How many elements fall into the given bin
                avg_conf = (
                    torch.sum(torch.stack([x[2] for x in filtered_tuples])) / len_bin
                )  # Avg confidence of bin
                accuracy = (torch.tensor(correct, dtype=torch.float32) / len_bin).to(
                    device
                )  # Accuracy of bin
            return accuracy, avg_conf, len_bin

        bin_size = torch.tensor(1.0 / bin_num)
        upper_bounds = torch.arange(bin_size, 1 + bin_size, bin_size)

        accuracies = []
        num_in_each_bin = []

        for conf_thresh in upper_bounds:
            acc, avg_conf, len_bin = compute_binned_acc_conf(
                conf_thresh - bin_size, conf_thresh, confidence, y_pred, y_true, device
            )
            accuracies.append(acc)
            num_in_each_bin.append(len_bin)

        acc_all = []
        for conf in confidence:
            idx = int(conf // (1 / bin_num))
            acc_all.append(accuracies[idx])

        return torch.stack(acc_all), torch.stack(num_in_each_bin)

    def calculate_cal_term(acc_vector, conf_vector, num_in_each_bin):
        bin_error = acc_vector * torch.log(conf_vector)
        cal_term = -torch.sum(bin_error)
        return cal_term

    probs = F.softmax(logits, dim=1)
    y_pred = torch.max(logits, axis=1)[1]
    confidence = torch.max(probs, axis=1)[0]
    acc_vector, num_in_each_bin = calculate_confidence_vec(
        confidence, y_pred, y_true, device
    )
    cal_term = calculate_cal_term(acc_vector, confidence, num_in_each_bin)

    lmbda = torch.tensor(lmbda)
    annealing_coef = torch.min(lmbda, torch.tensor(lmbda * (epoch + 1) / epochs))

    return cal_term * annealing_coef


def cal_eval_model(gnn, dataset, device, data_name, gnn_name, draw=True):
    data = dataset[0].to(device)
    with torch.no_grad():
        gnn.to(device)
        gnn.eval()
        test_logits = gnn(data)[data.test_mask]
        test_labels = data.y[data.test_mask]

    y_pred_test = test_logits.max(1)[1].detach().cpu()
    prob_pred_test = F.softmax(test_logits, dim=1)
    test_acc = (y_pred_test == test_labels).sum() / len(test_labels)

    a = np.array(np.arange(0, prob_pred_test.shape[0]))
    p_y = prob_pred_test[a, test_labels].detach().cpu().numpy()
    tiny = np.finfo(np.float).tiny  # To avoid division by 0 warning
    nll = -np.sum(np.log(p_y + tiny)) / prob_pred_test.shape[0]

    probs = list(prob_pred_test.detach().cpu().numpy())
    labels = list(test_labels.detach().cpu().numpy())
    ece = lower_bound_scaling_ce(probs, labels, p=1, debias=False, mode="top-label")
    marg_ece = lower_bound_scaling_ce(probs, labels, p=1, debias=False, mode="marginal")

    if draw:
        draw_RD(
            prob_pred_test.detach().cpu().numpy(),
            y_pred_test.numpy(),
            test_labels.detach().cpu().numpy(),
            data_name,
            gnn_name,
            ece,
        )
        draw_CH(
            prob_pred_test.detach().cpu().numpy(),
            y_pred_test.numpy(),
            test_labels.detach().cpu().numpy(),
            data_name,
            gnn_name,
            ece,
        )

    return ece, marg_ece, nll, test_acc


def draw_RD(prob_pred_test, y_pred_test, y_true_test, data_name, gnn_name, ece):
    confs_pred_test = np.max(prob_pred_test, axis=1)
    bin_info_uncal = get_uncalibrated_res(y_true_test, confs_pred_test, y_pred_test)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex="col", sharey="row")
    rel_diagram_sub(
        bin_info_uncal[0],
        bin_info_uncal[1],
        ax,
        15,
        "Reliability Diagram",
        "Confidence",
        "Accuracy",
        ece,
        data_name,
        gnn_name,
    )

    fig.savefig("output/figures/{}_{}_rd.png".format(gnn_name, data_name))


def draw_CH(prob_pred_test, y_pred_test, y_true_test, data_name, gnn_name, ece):
    confs_pred_test = np.max(prob_pred_test, axis=1)
    bin_info_uncal = get_uncalibrated_res(y_true_test, confs_pred_test, y_pred_test)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex="col", sharey="row")
    ave_conf, ave_acc = conf_histogram_sub(
        bin_info_uncal[0],
        bin_info_uncal[1],
        bin_info_uncal[2],
        ax,
        15,
        "Confidence Histogram",
        "Confidence",
        "Proportion of samples",
        ece,
        data_name,
        gnn_name,
    )
    fig.savefig("output/figures/{}_{}_ch.png".format(gnn_name, data_name))


def cal_method_eval(
    test_probs, test_labels, device, data_name, gnn_name, cal_method_name, draw=False
):
    test_probs, test_labels = (
        torch.tensor(test_probs).to(device),
        torch.tensor(test_labels).to(device),
    )
    prob_pred_test = test_probs

    a = np.array(np.arange(0, prob_pred_test.shape[0]))
    P_y = prob_pred_test[a, test_labels].detach().cpu().numpy()
    tiny = np.finfo(np.float).tiny  # To avoid division by 0 warning
    nll = -np.sum(np.log(P_y + tiny)) / prob_pred_test.shape[0]

    probs = list(prob_pred_test.detach().cpu().numpy())
    labels = list(test_labels.detach().cpu().numpy())
    ece = lower_bound_scaling_ce(probs, labels, p=1, debias=False, mode="top-label")
    marg_ece = lower_bound_scaling_ce(probs, labels, p=1, debias=False, mode="marginal")

    y_pred_test = test_probs.max(1)[1]
    cal_test_acc = (y_pred_test == test_labels).sum() / len(test_labels)

    if draw:
        draw_RD_cal(
            prob_pred_test.detach().cpu().numpy(),
            y_pred_test,
            test_labels.detach().cpu().numpy(),
            data_name,
            gnn_name,
            ece,
            cal_method_name,
        )
        draw_CH_cal(
            prob_pred_test.detach().cpu().numpy(),
            y_pred_test,
            test_labels.detach().cpu().numpy(),
            data_name,
            gnn_name,
            ece,
            cal_method_name,
        )

    return ece, marg_ece, nll, cal_test_acc.item()


def draw_RD_cal(
    prob_pred_test, y_pred_test, y_true_test, data_name, gnn_name, ece, cal_method_name
):
    confs_pred_test = np.max(prob_pred_test, axis=1)
    bin_info_uncal = get_uncalibrated_res(y_true_test, confs_pred_test, y_pred_test)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex="col", sharey="row")
    rel_diagram_sub_cal(
        bin_info_uncal[0],
        bin_info_uncal[1],
        ax,
        15,
        "Reliability Diagram",
        "Confidence",
        "Accuracy",
        ece,
        data_name,
        gnn_name,
        1,
    )
    fig.savefig(
        "output/figures/{}_{}_{}_rd.png".format(gnn_name, data_name, cal_method_name)
    )


def draw_CH_cal(
    prob_pred_test, y_pred_test, y_true_test, data_name, gnn_name, ece, cal_method_name
):
    confs_pred_test = np.max(prob_pred_test, axis=1)
    bin_info_uncal = get_uncalibrated_res(y_true_test, confs_pred_test, y_pred_test)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex="col", sharey="row")
    ave_conf, ave_acc = conf_histogram_sub_cal(
        bin_info_uncal[0],
        bin_info_uncal[1],
        bin_info_uncal[2],
        ax,
        15,
        "Confidence Histogram",
        "Confidence",
        "Proportion of samples",
        ece,
        data_name,
        gnn_name,
        1,
    )
    fig.savefig(
        "output/figures/{}_{}_{}_ch.png".format(gnn_name, data_name, cal_method_name)
    )


def produce_logits(gnn, data, device):
    test_logits_list = []
    test_labels_list = []
    gnn.to(device)
    data = data.to(device)

    with torch.no_grad():
        gnn.eval()
        logits = gnn(data)[data.test_mask]
        test_logits_list.append(logits)
        test_labels_list.append(data.y[data.test_mask])
        test_logits = torch.cat(test_logits_list).to(device)
        test_labels = torch.cat(test_labels_list).to(device)
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        gnn.eval()
        logits = gnn(data)[data.val_mask]
        val_logits_list.append(logits)
        val_labels_list.append(data.y[data.val_mask])
        val_logits = torch.cat(val_logits_list).to(device)
        val_labels = torch.cat(val_labels_list).to(device)
    val_probs = F.softmax(val_logits, dim=1).detach().cpu().numpy()
    val_labels = val_labels.detach().cpu().numpy()
    test_probs = F.softmax(test_logits, dim=1).detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    val_logits, test_logits = (
        val_logits.detach().cpu().numpy(),
        test_logits.detach().cpu().numpy(),
    )
    with torch.no_grad():
        gnn.eval()
        logits = gnn(data)
    logits_list = []
    logits_list.append(logits)
    logits = torch.cat(logits_list).to(device)
    probs = F.softmax(logits, 1).detach().cpu().numpy()

    return (
        val_probs,
        test_probs,
        val_logits,
        test_logits,
        val_labels,
        test_labels,
        logits.detach().cpu().numpy(),
        probs,
    )


def get_uncalibrated_res(y_true, confs_pred, y_pred, M=15):
    bin_size = 1 / M
    return get_bin_info(confs_pred, y_pred, y_true, bin_size=bin_size)


def get_bin_info(conf, pred, true, bin_size):
    """
    Get accuracy, confidence and elements in bin information for all the bins.
    """
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true
        )
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    Computes accuracy and average confidence for bin.
    """
    filtered_tuples = [
        x
        for x in zip(pred, true, conf)
        if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
    ]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len(
            [x for x in filtered_tuples if x[0] == x[1]]
        )  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements fall into given bin
        avg_conf = (
            sum([x[2] for x in filtered_tuples]) / len_bin
        )  # Avg confidence of BIN
        accuracy = float(correct) / len_bin
        return accuracy, avg_conf, len_bin


def rel_diagram_sub(
    accs,
    confs,
    ax,
    M=15,
    name="Reliability Diagram",
    xname="",
    yname="",
    ece=None,
    data_name=None,
    gnn_name=None,
):
    plt.plot(
        np.arange(0, 1, 0.1), np.arange(0, 1, 0.1), linestyle="dashed", color="black"
    )

    bin_size = 1 / M
    # Center of each bin
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)
    outputs = np.array(confs)
    acc = np.array(accs)
    # Bars with outputs
    output_plt = ax.bar(
        positions, acc, width=bin_size, edgecolor="black", color="blue", zorder=0
    )
    # Plot gap first, so its below everything
    gap_plt = ax.bar(
        positions,
        outputs - acc,
        bottom=acc,
        width=bin_size,
        edgecolor="red",
        hatch="/",
        color="red",
        alpha=0.3,
        linewidth=2,
        label="Gap",
        zorder=3,
    )
    ax.text(
        0.55,
        0.1,
        "ECE = {}%".format(round(ece * 100, 1)),
        size=14,
        backgroundcolor="grey",
    )

    # Line plot with center line.
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.legend(handles=[gap_plt, output_plt])
    ax.legend(loc=2, prop={"size": 14})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("{} on {}".format(gnn_name.upper(), data_name), fontsize=14)
    ax.set_xlabel(xname, fontsize=14, color="black")
    ax.set_ylabel(yname, fontsize=14, color="black")
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)


def conf_histogram_sub(
    accs,
    confs,
    nums,
    ax,
    M=15,
    name="Reliability Diagram",
    xname="",
    yname="",
    ece=None,
    data_name=None,
    gnn_name=None,
):
    acc = np.array(accs)
    conf = np.array(confs)
    num = np.array(nums) / np.array(nums).sum()
    bin_size = 1 / M
    # Center of each bin
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)
    # Bars with nums
    output_plt = ax.bar(
        positions, num, width=bin_size, edgecolor="black", color="blue", zorder=0
    )
    ave_conf = (conf * num).sum()
    ax.plot(
        [ave_conf, ave_conf], [0, 1], linestyle="-", color="blue", label="Avg. conf."
    )
    ave_acc = (acc * num).sum()
    ax.plot([ave_acc, ave_acc], [0, 1], linestyle="--", color="red", label="Acc.")
    ax.legend(loc=2, prop={"size": 14})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("{} on {}".format(gnn_name.upper(), data_name), fontsize=14)
    ax.set_xlabel(xname, fontsize=14, color="black")
    ax.set_ylabel(yname, fontsize=14, color="black")
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    return ave_conf, ave_acc


def rel_diagram_sub_cal(
    accs,
    confs,
    ax,
    M=15,
    name="Reliability Diagram",
    xname="",
    yname="",
    ece=None,
    data_name=None,
    gnn_name=None,
    cal=None,
):
    plt.plot(
        np.arange(0, 1, 0.1), np.arange(0, 1, 0.1), linestyle="dashed", color="black"
    )
    bin_size = 1 / M
    # Cennter of each bin
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)
    outputs = np.array(confs)
    acc = np.array(accs)
    # Bars with outputs
    output_plt = ax.bar(
        positions, acc, width=bin_size, edgecolor="black", color="blue", zorder=0
    )
    # Plot gap first, so its below everything
    gap_plt = ax.bar(
        positions,
        outputs - acc,
        bottom=acc,
        width=bin_size,
        edgecolor="red",
        hatch="/",
        color="red",
        alpha=0.3,
        linewidth=2,
        label="Gap",
        zorder=3,
    )
    ax.text(
        0.55,
        0.1,
        "ECE = {}%".format(round(ece * 100, 1)),
        size=14,
        backgroundcolor="grey",
    )

    # Line plot with center line.
    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.legend(handles=[gap_plt, output_plt])
    ax.legend(loc=2, prop={"size": 14})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if cal == None:
        ax.set_title("{} on {}".format(gnn_name.upper(), data_name), fontsize=14)
    else:
        ax.set_title("{} on {} (cal.)".format(gnn_name.upper(), data_name), fontsize=14)
    ax.set_xlabel(xname, fontsize=14, color="black")
    ax.set_ylabel(yname, fontsize=14, color="black")
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)


def conf_histogram_sub_cal(
    accs,
    confs,
    nums,
    ax,
    M=15,
    name="Reliability Diagram",
    xname="",
    yname="",
    ECE=None,
    data_name=None,
    gnn_name=None,
    cal=None,
):
    acc = np.array(accs)
    conf = np.array(confs)
    num = np.array(nums) / np.array(nums).sum()
    bin_size = 1 / M
    # Center of each bin
    positions = np.arange(0 + bin_size / 2, 1 + bin_size / 2, bin_size)
    # Bars with nums
    output_plt = ax.bar(
        positions, num, width=bin_size, edgecolor="black", color="blue", zorder=0
    )
    ave_conf = (conf * num).sum()
    ax.plot(
        [ave_conf, ave_conf], [0, 1], linestyle="-", color="blue", label="Avg. conf."
    )
    ave_acc = (acc * num).sum()
    ax.plot([ave_acc, ave_acc], [0, 1], linestyle="--", color="red", label="Acc.")
    ax.legend(loc=2, prop={"size": 14})

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if cal == None:
        ax.set_title("{} on {}".format(gnn_name.upper(), data_name), fontsize=14)
    else:
        ax.set_title("{} on {} (cal.)".format(gnn_name.upper(), data_name), fontsize=14)
    ax.set_xlabel(xname, fontsize=14, color="black")
    ax.set_ylabel(yname, fontsize=14, color="black")
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    return ave_conf, ave_acc


def cal_loss_rmse(
    y_true, logits, lmbda, epoch, epochs, device,
):
    def calculate_confidence_vec(confidence, y_pred, y_true, device, bin_num=15):
        def compute_binned_acc_conf(
            conf_thresh_lower, conf_thresh_upper, conf, pred, true, device
        ):

            filtered_tuples = [
                x
                for x in zip(pred, true, conf)
                if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
            ]
            if len(filtered_tuples) < 1:
                return (
                    torch.tensor(0.0).to(device),
                    torch.tensor(0.0, requires_grad=True).to(device),
                    torch.tensor(0).to(device),
                )
            else:
                correct = len(
                    [x for x in filtered_tuples if x[0] == x[1]]
                )  # How many correct labels
                len_bin = torch.tensor(len(filtered_tuples)).to(
                    device
                )  # How many elements falls into given bin
                avg_conf = (
                    torch.sum(torch.stack([x[2] for x in filtered_tuples])) / len_bin
                )  # Avg confidence of BIN
                accuracy = (torch.tensor(correct, dtype=torch.float32) / len_bin).to(
                    device
                )  # Accuracy of bin
            return accuracy, avg_conf, len_bin

        bin_size = torch.tensor(1.0 / bin_num)
        upper_bounds = torch.arange(bin_size, 1 + bin_size, bin_size)

        accuracies = []
        confidences = []
        num_in_each_bin = []

        for conf_thresh in upper_bounds:
            acc, avg_conf, len_bin = compute_binned_acc_conf(
                conf_thresh - bin_size, conf_thresh, confidence, y_pred, y_true, device
            )
            accuracies.append(acc)
            confidences.append(avg_conf)
            num_in_each_bin.append(len_bin)
        return (
            torch.stack(accuracies),
            torch.stack(confidences),
            torch.stack(num_in_each_bin),
        )

    def calculate_cal_term(acc_vector, conf_vector, num_in_each_bin, y_true):
        power = 2
        bin_error = abs(acc_vector - conf_vector) ** power
        bin_probs = num_in_each_bin / y_true.shape[0]
        ece_score = torch.sum((bin_error * bin_probs)) ** (1 / power)
        return cal_term

    probs = F.softmax(logits, dim=1)
    y_pred = torch.max(logits, axis=1)[1]
    confidence = torch.max(probs, axis=1)[0]
    acc_vector, conf_vector, num_in_each_bin = calculate_confidence_vec(
        confidence, y_pred, y_true, device
    )
    cal_term = calculate_cal_term(acc_vector, conf_vector, num_in_each_bin, y_true)

    lmbda = torch.tensor(lmbda)
    annealing_coef = torch.min(lmbda, torch.tensor(lmbda * (epoch + 1) / epochs))

    return cal_term * annealing_coef
