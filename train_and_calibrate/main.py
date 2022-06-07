import os
import numpy as np
from tqdm import tqdm
import torch

from args import get_args
from get_dataset import get_planetoid_dataset
from create_model import (
    create_gcn_model,
    create_gat_model,
    create_sgc_model,
    create_gfnn_model,
    create_appnp_model,
)
from tuning import search_best_wd
from utils import (
    reproducibility_seed,
    training,
    cal_eval_model,
    cal_method_eval,
    produce_logits,
)
from calibration_methods import (
    HistogramBinning,
    IsotonicRegression,
    BayesianBinningQuantiles,
    TemperatureScaling,
    RBS,
)
from metacal.metacal import MetaCalMisCoverage


def train_gnn(dataset, gnn, optimizer, args):
    if args.tune_wd == True:
        print("Tune weight decay...")
        weight_decay = search_best_wd(dataset, gnn, optimizer, args)
        optimizer.param_groups[0]["weight_decay"] = weight_decay

    results, results_val = [], []

    for num_run in tqdm(range(args.num_runs)):
        test_acc, val_acc = training(
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
        results.append([test_acc])
        results_val.append([val_acc])
    test_acc_mean = np.mean(results, axis=0)[0] * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    val_acc_mean = np.mean(results_val, axis=0)[0] * 100
    val_acc_std = np.sqrt(np.var(results_val, axis=0)[0]) * 100

    print(
        "Test accuracy is: {}% \u00B1 {}%".format(
            np.round(test_acc_mean, 2), np.round(test_acc_std, 2)
        )
    )
    print(
        "Val accuracy is: {}% \u00B1 {}%".format(
            np.round(val_acc_mean, 2), np.round(val_acc_std, 2)
        )
    )


def calibrate_gnn(dataset, gnn, args):
    data = dataset[0]
    res_uncal, res_his, res_iso, res_bbq, res_ts, res_meta, res_rbs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    checkpoints_path = "checkpoints/{}_{}/".format(args.model, args.dataset)

    for num_run in tqdm(range(args.num_runs)):
        save_path = checkpoints_path + str(num_run)
        gnn.load_state_dict(torch.load(save_path))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gnn.to(device)

        # Uncal
        ece, marg_ece, nll, test_acc = cal_eval_model(
            gnn, dataset, device, args.dataset, args.model
        )
        res_uncal.append([ece, marg_ece, nll, test_acc])

        # Calibrate
        (
            val_probs,
            test_probs,
            val_logits,
            test_logits,
            val_labels,
            test_labels,
            logits,
            probs,
        ) = produce_logits(gnn, data, device)

        # His
        his_model = HistogramBinning()
        his_model.fit(val_probs, val_labels)
        cal_probs_test = his_model.predict_proba(test_probs)
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "his",
            draw=False,
        )
        res_his.append([ece, marg_ece, nll, cal_test_acc])

        # Iso
        iso_model = IsotonicRegression()
        iso_model.fit(val_probs, val_labels)
        cal_probs_test = iso_model.predict_proba(test_probs)
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "iso",
            draw=False,
        )
        res_iso.append([ece, marg_ece, nll, cal_test_acc])

        # BBQ
        BBQ_model = BayesianBinningQuantiles()
        BBQ_model.fit(val_probs, val_labels)
        cal_probs_test = BBQ_model.predict_proba(test_probs)
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "bbq",
            draw=False,
        )
        res_bbq.append([ece, marg_ece, nll, cal_test_acc])

        # TS
        TS_model = TemperatureScaling()
        TS_model.fit(val_logits, val_labels)
        cal_probs_test = TS_model.predict_proba(test_logits)
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "ts",
            draw=False,
        )
        res_ts.append([ece, marg_ece, nll, cal_test_acc])

        # Meta
        a = 0.05
        meta_model = MetaCalMisCoverage(a)
        meta_model.fit(val_logits, val_labels)
        cal_probs_test = meta_model.predict(test_logits)
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "meta",
            draw=False,
        )
        res_meta.append([ece, marg_ece, nll, cal_test_acc])

        # RBS
        cal_probs_test = RBS(
            data, probs, val_logits, val_labels, test_logits, args.num_bins_rbs
        )
        ece, marg_ece, nll, cal_test_acc = cal_method_eval(
            cal_probs_test,
            test_labels,
            device,
            args.dataset,
            args.model,
            "rbs",
            draw=False,
        )
        res_rbs.append([ece, marg_ece, nll, cal_test_acc])

    # Average
    cal_name_list = ["Uncal", "His", "Iso", "BBQ", "TS", "Meta", "RBS"]
    cal_results_map = {
        "Uncal": res_uncal,
        "His": res_his,
        "Iso": res_iso,
        "BBQ": res_bbq,
        "TS": res_ts,
        "Meta": res_meta,
        "RBS": res_rbs,
    }
    cal_results_list = [cal_results_map[x] for x in cal_name_list]
    for result, cal_method in zip(cal_results_list, cal_name_list):
        # 0: mean, 1: std
        ece0, marg_ece0, nll0, cal_test_acc0 = np.mean(result, axis=0) * 100
        nll0 = nll0 / 100
        ece1, marg_ece1, nll1, cal_test_acc1 = (
            np.sqrt(np.var(result, axis=0)[0]) * 100,
            np.sqrt(np.var(result, axis=0)[1]) * 100,
            np.sqrt(np.var(result, axis=0)[2]),
            np.sqrt(np.var(result, axis=0)[3]) * 100,
        )

        path = "output/{}_{}_{}.txt".format(args.model, args.dataset, args.num_runs)
        with open(path, "a") as f:
            f.write(cal_method)
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
                    np.round(cal_test_acc0, 2), np.round(cal_test_acc1, 2)
                )
            )
            f.write("\n")
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    print(args)
    dataset = get_planetoid_dataset(args.dataset)
    if not os.path.exists("output/"):
        os.makedirs("output/")
    if not os.path.exists("output/figures/"):
        os.makedirs("output/figures/")

    print("Train...")
    reproducibility_seed(0, 0)
    if args.model == "gcn":
        gnn, optimizer = create_gcn_model(dataset)
    elif args.model == "gat":
        gnn, optimizer = create_gat_model(dataset)
    elif args.model == "sgc":
        gnn, optimizer = create_sgc_model(dataset)
    elif args.model == "gfnn":
        gnn, optimizer = create_gfnn_model(dataset)
    elif args.model == "appnp":
        gnn, optimizer = create_appnp_model(dataset)
    train_gnn(dataset, gnn, optimizer, args)

    print("Calibrate...")
    reproducibility_seed(0, 0)
    calibrate_gnn(dataset, gnn, args)
