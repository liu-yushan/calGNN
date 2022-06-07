import numpy as np
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK
from math import log
from torch.optim import Adam

from utils import training


def objective_wd(dataset, gnn, optimizer, args, space):
    optimizer0 = Adam(gnn.parameters(), lr=0.2, weight_decay=space["weight_decay"])
    evals = training(
        dataset,
        gnn,
        optimizer0,
        args.model,
        args.dataset,
        args.epochs,
        args.early_stopping,
        args.patience,
        args.add_cal_loss,
        args.alpha,
        args.lmbda,
        num_run=-1,
        save_model=False,
    )
    return {"loss": -evals[-1], "status": STATUS_OK}


def search_best_wd(dataset, gnn, optimizer, args):
    f = partial(objective_wd, dataset, gnn, optimizer, args)
    space = {"weight_decay": hp.loguniform("weight_decay", log(1e-9), log(1e-3))}
    best = fmin(f, space=space, algo=tpe.suggest, max_evals=args.max_search, rstate=np.random.default_rng(0))
    print("Best weight decay: ", best["weight_decay"])
    return best["weight_decay"]
