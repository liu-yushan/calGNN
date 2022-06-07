# On Calibration of Graph Neural Networks for Node Classification

This repository contains the code for the paper [On Calibration of Graph Neural Networks for Node Classification](https://arxiv.org/abs/2206.01570).


<h3> How to run </h3>

The dependencies required to run the code are specified in [`pyproject.toml`](https://github.com/liu-yushan/calGNN/blob/main/pyproject.toml). Run `poetry install` to install the dependencies from [`poetry.lock`](https://github.com/liu-yushan/calGNN/blob/main/poetry.lock), which lists the exact versions used. For more information about Poetry, a tool for dependency management and packaging in Python, see https://python-poetry.org/docs/.

The commands for running the experiments from the paper can be found in `run.txt` in the corresponding folders.


<h3> Arguments </h3> 

Defined in `args.py`.

`--dataset`: str. Dataset name.

`--model`: str. Model name.

`--num_runs`: int. Number of independent runs for each model.

`--epochs`: int. Number of training epochs.

`--early_stopping`: bool. Whether early stopping is applied. Note that if no early stopping is used, do not pass this argument to the command line (also not `--early_stopping False`) since almost anything will be interpreted as `True` by Python. The same holds for the other boolean arguments.

`--patience`: int. Number of epochs before doing early stopping.

`--tune_wd`: bool. Whether a hyperparameter search is conducted for the weight decay value.

`--max_search`: int. Number of searches for weight decay tuning.

`--add_cal_loss`: bool. Whether the calibration loss term is added.

`--alpha`: float. Hyperparameter for the calibration loss.

`--lmbda`: float. Hyperparameter for the calibration loss.

`--num_bins_rbs`: int. Number of bins used for the calibration method RBS.
