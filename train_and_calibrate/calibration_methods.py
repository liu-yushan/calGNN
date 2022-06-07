# Code partly from https://github.com/JonathanWenger/pycalib/blob/master/pycalib/calibration_methods.py

import torch
from torch import nn, optim
from torch_geometric.utils import to_networkx
import scipy as sp

# Standard library imports
import numpy as np
import numpy.matlib
import warnings
import matplotlib.pyplot as plt

# SciPy imports
import scipy.stats
import scipy.optimize
import scipy.special
import scipy.cluster.vq

# Scikit learn imports
import sklearn
import sklearn.multiclass
import sklearn.utils
from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed

# Calibration models
import sklearn.isotonic
import sklearn.linear_model


# Ignore binned_statistic FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


class CalibrationMethod(sklearn.base.BaseEstimator):
    """
    A generic class for probability calibration
    A calibration method takes a set of posterior class probabilities and transform them into calibrated posterior
    probabilities. Calibrated in this sense means that the empirical frequency of a correct class prediction matches its
    predicted posterior probability.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.
        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def predict(self, X):
        """
        Predict the class of new samples after scaling. Predictions are identical to the ones from the uncalibrated
        classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.
        Returns
        -------
        C : array, shape (n_samples,)
            The predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)


class OneVsRestCalibrator(sklearn.base.BaseEstimator):
    """One-vs-the-rest (OvR) multiclass strategy
    Also known as one-vs-all, this strategy consists in fitting one calibrator
    per class. The probabilities to be calibrated of the other classes are summed.
    For each calibrator, the class is fitted against all the other classes.
    Parameters
    ----------
    calibrator : CalibrationMethod object
        A CalibrationMethod object implementing `fit` and `predict_proba`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details.
    Attributes
    ----------
    calibrators_ : list of `n_classes` estimators
        Estimators used for predictions.
    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    """

    def __init__(self, calibrator, n_jobs=None):
        self.calibrator = calibrator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Calibration data.
        y : (sparse) array-like, shape = [n_samples, ]
            Multi-class labels.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.calibrators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(OneVsRestCalibrator._fit_binary)(
                self.calibrator,
                X,
                column,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )
        return self

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, ["classes_", "calibrators_"])

        # Y[i, j] gives the probability that sample i has the label j.
        Y = np.array(
            [
                c.predict_proba(
                    np.column_stack(
                        [
                            np.sum(np.delete(X, obj=i, axis=1), axis=1),
                            X[:, self.classes_[i]],
                        ]
                    )
                )[:, 1]
                for i, c in enumerate(self.calibrators_)
            ]
        ).T

        if len(self.calibrators_) == 1:
            # Only one estimator, but we still want to return probabilities for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        # Pad with zeros for classes not in training data
        if np.shape(Y)[1] != np.shape(X)[1]:
            p_pred = np.zeros(np.shape(X))
            p_pred[:, self.classes_] = Y
            Y = p_pred

        # Normalize probabilities to 1.
        Y = sklearn.preprocessing.normalize(
            Y, norm="l1", axis=1, copy=True, return_norm=False
        )
        return np.clip(Y, a_min=0, a_max=1)

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _first_calibrator(self):
        return self.calibrators_[0]

    @staticmethod
    def _fit_binary(calibrator, X, y, classes=None):
        """
        Fit a single binary calibrator.
        Parameters
        ----------
        calibrator
        X
        y
        classes
        Returns
        -------
        """
        # Sum probabilities of combined classes in calibration training data X
        cl = classes[1]
        X = np.column_stack([np.sum(np.delete(X, cl, axis=1), axis=1), X[:, cl]])

        # Check whether only one label is present in training data
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            if classes is not None:
                if y[0] == -1:
                    c = 0
                else:
                    c = y[0]
                warnings.warn(
                    "Label %s is present in all training examples." % str(classes[c])
                )
            calibrator = _ConstantCalibrator().fit(X, unique_y)
        else:
            calibrator = clone(calibrator)
            calibrator.fit(X, y)
        return calibrator


class HistogramBinning(CalibrationMethod):
    def __init__(self, mode="equal_freq", n_bins=20, input_range=[0, 1]):
        super().__init__()
        if mode in ["equal_width", "equal_freq"]:
            self.mode = mode
        else:
            raise ValueError(
                "Mode not recognized. Choose on of 'equal_width', or 'equal_freq'."
            )
        self.n_bins = n_bins
        self.input_range = input_range

    def fit(self, X, y, n_jobs=None):
        if X.ndim == 1:
            raise ValueError(
                "Calibration training data must have shape (n_samples, n_classes)."
            )
        elif np.shape(X)[1] == 2:
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(
                calibrator=clone(self), n_jobs=n_jobs
            )
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def _fit_binary(self, X, y):
        if self.mode == "equal_width":
            # Compute probability of class 1 in each equal width bin
            binned_stat = scipy.stats.binned_statistic(
                x=X[:, 1],
                values=np.equal(1, y),
                statistic="mean",
                bins=self.n_bins,
                range=self.input_range,
            )
            self.prob_class_1 = binned_stat.statistic
            self.binning = binned_stat.bin_edges
        elif self.mode == "equal_freq":
            # Find binning based on equal frequency
            self.binning = np.quantile(
                X[:, 1],
                q=np.linspace(
                    self.input_range[0], self.input_range[1], self.n_bins + 1
                ),
            )

            # Compute probability of class 1 in equal frequency bins
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = (
                len(self.binning) - 1
            )  # include rightmost edge in partition
            self.prob_class_1 = [
                y[digitized == i].mean() for i in range(1, len(self.binning))
            ]

        return self

    def predict_proba(self, X):
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binning", "prob_class_1"])
            # Find bin of predictions
            digitized = np.digitize(X[:, 1], bins=self.binning)
            digitized[digitized == len(self.binning)] = (
                len(self.binning) - 1
            )  # include rightmost edge in partition
            # Transform to empirical frequency of class 1 in each bin
            p1 = np.array([self.prob_class_1[j] for j in (digitized - 1)])
            # If empirical frequency is NaN, do not change prediction
            p1 = np.where(np.isfinite(p1), p1, X[:, 1])
            assert np.all(np.isfinite(p1)), "Predictions are not all finite."

            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class IsotonicRegression(CalibrationMethod):
    def __init__(self, out_of_bounds="clip"):
        super().__init__()
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, n_jobs=None):
        """
        Fit the calibration method based on the given uncalibrated class probabilities X and ground truth labels y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        n_jobs : int or None, optional (default=None)
            The number of jobs to use for the computation.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if X.ndim == 1:
            raise ValueError(
                "Calibration training data must have shape (n_samples, n_classes)."
            )
        elif np.shape(X)[1] == 2:
            self.isotonic_regressor_ = sklearn.isotonic.IsotonicRegression(
                increasing=True, out_of_bounds=self.out_of_bounds
            )
            self.isotonic_regressor_.fit(X[:, 1], y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(
                calibrator=clone(self), n_jobs=n_jobs
            )
            self.onevsrest_calibrator_.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.

        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, "isotonic_regressor_")
            p1 = self.isotonic_regressor_.predict(X[:, 1])
            return np.column_stack([1 - p1, p1])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class BayesianBinningQuantiles(CalibrationMethod):
    def __init__(self, C=10, input_range=[0, 1]):
        super().__init__()
        self.C = C
        self.input_range = input_range

    def _binning_model_logscore(self, probs, y, partition, N_prime=2):
        # Setup
        B = len(partition) - 1
        p = (partition[1:] - partition[:-1]) / 2 + partition[:-1]

        # Compute positive and negative samples in given bins
        N = np.histogram(probs, bins=partition)[0]

        digitized = np.digitize(probs, bins=partition)
        digitized[digitized == len(partition)] = (
            len(partition) - 1
        )  # include rightmost edge in partition
        m = [y[digitized == i].sum() for i in range(1, len(partition))]
        n = N - m

        # Compute the parameters of the Beta priors
        tiny = np.finfo(
            np.float
        ).tiny  # Avoid scipy.special.gammaln(0), which can arise if bin has zero width
        alpha = N_prime / B * p
        alpha[alpha == 0] = tiny
        beta = N_prime / B * (1 - p)
        beta[beta == 0] = tiny

        # Prior for a given binning model (uniform)
        log_prior = -np.log(self.T)

        # Compute the marginal log-likelihood for the given binning model
        log_likelihood = np.sum(
            scipy.special.gammaln(N_prime / B)
            + scipy.special.gammaln(m + alpha)
            + scipy.special.gammaln(n + beta)
            - (
                scipy.special.gammaln(N + N_prime / B)
                + scipy.special.gammaln(alpha)
                + scipy.special.gammaln(beta)
            )
        )

        # Compute score for the given binning model
        log_score = log_prior + log_likelihood
        return log_score

    def fit(self, X, y, n_jobs=None):
        if X.ndim == 1:
            raise ValueError(
                "Calibration training data must have shape (n_samples, n_classes)."
            )
        elif np.shape(X)[1] == 2:
            self.binnings = []
            self.log_scores = []
            self.prob_class_1 = []
            self.T = 0
            return self._fit_binary(X, y)
        elif np.shape(X)[1] > 2:
            self.onevsrest_calibrator_ = OneVsRestCalibrator(
                calibrator=clone(self), n_jobs=n_jobs
            )
            self.onevsrest_calibrator_.fit(X, y)
            return self

    def _fit_binary(self, X, y):
        # Determine number of bins
        N = len(y)
        min_bins = int(max(1, np.floor(N ** (1 / 3) / self.C)))
        max_bins = int(min(np.ceil(N / 5), np.ceil(self.C * N ** (1 / 3))))
        self.T = max_bins - min_bins + 1

        # Define (equal frequency) binning models and compute scores
        self.binnings = []
        self.log_scores = []
        self.prob_class_1 = []
        for i, n_bins in enumerate(range(min_bins, max_bins + 1)):
            # Compute binning from data and set outer edges to range
            binning_tmp = np.quantile(
                X[:, 1],
                q=np.linspace(self.input_range[0], self.input_range[1], n_bins + 1),
            )
            binning_tmp[0] = self.input_range[0]
            binning_tmp[-1] = self.input_range[1]
            # Enforce monotonicity of binning (np.quantile does not guarantee monotonicity)
            self.binnings.append(np.maximum.accumulate(binning_tmp))
            # Compute score
            self.log_scores.append(
                self._binning_model_logscore(
                    probs=X[:, 1], y=y, partition=self.binnings[i]
                )
            )

            # Compute empirical accuracy for all bins
            digitized = np.digitize(X[:, 1], bins=self.binnings[i])
            # include rightmost edge in partition
            digitized[digitized == len(self.binnings[i])] = len(self.binnings[i]) - 1

            def empty_safe_bin_mean(a, empty_value):
                """
                Assign the bin mean to an empty bin. Corresponds to prior assumption of the underlying classifier
                being calibrated.
                """
                if a.size == 0:
                    return empty_value
                else:
                    return a.mean()

            self.prob_class_1.append(
                [
                    empty_safe_bin_mean(
                        y[digitized == k],
                        empty_value=(self.binnings[i][k] + self.binnings[i][k - 1]) / 2,
                    )
                    for k in range(1, len(self.binnings[i]))
                ]
            )

        return self

    def predict_proba(self, X):
        if X.ndim == 1:
            raise ValueError("Calibration data must have shape (n_samples, n_classes).")
        elif np.shape(X)[1] == 2:
            check_is_fitted(self, ["binnings", "log_scores", "prob_class_1", "T"])

            # Find bin for all binnings and the associated empirical accuracy
            posterior_prob_binnings = np.zeros(
                shape=[np.shape(X)[0], len(self.binnings)]
            )
            for i, binning in enumerate(self.binnings):
                bin_ids = np.searchsorted(binning, X[:, 1])
                bin_ids = np.clip(
                    bin_ids, a_min=0, a_max=len(binning) - 1
                )  # necessary if X is out of range
                posterior_prob_binnings[:, i] = [
                    self.prob_class_1[i][j] for j in (bin_ids - 1)
                ]

            # Computed score-weighted average
            norm_weights = np.exp(
                np.array(self.log_scores) - scipy.special.logsumexp(self.log_scores)
            )
            posterior_prob = np.sum(posterior_prob_binnings * norm_weights, axis=1)

            # Compute probability for other class
            return np.column_stack([1 - posterior_prob, posterior_prob])
        elif np.shape(X)[1] > 2:
            check_is_fitted(self, "onevsrest_calibrator_")
            return self.onevsrest_calibrator_.predict_proba(X)


class TemperatureScaling(CalibrationMethod):
    def __init__(self, T_init=1, verbose=False):
        super().__init__()
        if T_init <= 0:
            raise ValueError("Temperature not greater than 0.")
        self.T_init = T_init
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the calibration method based on the given uncalibrated class probabilities or logits X and ground truth
        labels y.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            Training data, i.e. predicted probabilities or logits of the base classifier on the calibration set.
        y : array-like, shape (n_samples,)
            Target classes.
        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # Define objective function (NLL / cross entropy)
        def objective(T):
            # Calibrate with given T
            P = scipy.special.softmax(X / T, axis=1)

            # Compute negative log-likelihood
            P_y = P[np.array(np.arange(0, X.shape[0])), y]
            tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
            NLL = -np.sum(np.log(P_y + tiny))
            return NLL

        # Derivative of the objective with respect to the temperature T
        def gradient(T):
            # Exponential terms
            E = np.exp(X / T)

            # Gradient
            dT_i = (
                np.sum(
                    E * (X - X[np.array(np.arange(0, X.shape[0])), y].reshape(-1, 1)),
                    axis=1,
                )
            ) / np.sum(E, axis=1)
            grad = -dT_i.sum() / T ** 2
            return grad

        # Optimize
        self.T = scipy.optimize.fmin_bfgs(
            f=objective, x0=self.T_init, fprime=gradient, gtol=1e-06, disp=self.verbose
        )[0]

        # Check for T > 0
        if self.T <= 0:
            raise ValueError("Temperature not greater than 0.")

        return self

    def predict_proba(self, X):
        """
        Compute calibrated posterior probabilities for a given array of posterior probabilities from an arbitrary
        classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_classes)
            The uncalibrated posterior probabilities.
        Returns
        -------
        P : array, shape (n_samples, n_classes)
            The predicted probabilities.
        """
        # Check is fitted
        check_is_fitted(self, "T")
        # Transform with scaled softmax
        return scipy.special.softmax(X / self.T, axis=1)

    def latent(self, z):
        """
        Evaluate the latent function Tz of temperature scaling.
        Parameters
        ----------
        z : array-like, shape=(n_evaluations,)
            Input confidence for which to evaluate the latent function.
        Returns
        -------
        f : array-like, shape=(n_evaluations,)
            Values of the latent function at z.
        """
        check_is_fitted(self, "T")
        return self.T * z


class TemperatureScaling_bins(CalibrationMethod):
    def __init__(self, T_init=1, verbose=False):
        super().__init__()
        if T_init <= 0:
            raise ValueError("Temperature not greater than 0.")
        self.T_init = T_init
        self.verbose = verbose

    def fit(self, X, y):

        # Define objective function (NLL / cross entropy)
        def objective(T):
            # Calibrate with given T
            P = scipy.special.softmax(X / T, axis=1)

            # Compute negative log-likelihood
            P_y = P[np.array(np.arange(0, X.shape[0])), y]
            tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
            NLL = -np.sum(np.log(P_y + tiny))
            return NLL

        # Derivative of the objective with respect to the temperature T
        def gradient(T):
            # Exponential terms
            E = np.exp(X / T)

            # Gradient
            dT_i = (
                np.sum(
                    E * (X - X[np.array(np.arange(0, X.shape[0])), y].reshape(-1, 1)),
                    axis=1,
                )
            ) / np.sum(E, axis=1)
            grad = -dT_i.sum() / T ** 2
            return grad

        # Optimize
        self.T = scipy.optimize.fmin_bfgs(
            f=objective, x0=self.T_init, fprime=gradient, gtol=1e-06, disp=self.verbose
        )[0]

        # Check for T > 0
        if self.T <= 0:
            raise ValueError("Temperature not greater than 0.")

        return self, self.T

    def predict_proba(self, X):

        # Check is fitted
        check_is_fitted(self, "T")
        # Transform with scaled softmax
        return scipy.special.softmax(X / self.T, axis=1)


class bin_mask_eqdis(nn.Module):
    def __init__(self, num_bins, sm_vector):
        super(bin_mask_eqdis, self).__init__()
        if not torch.is_tensor(sm_vector):
            self.sm_vector = torch.tensor(sm_vector)
        if torch.cuda.is_available():
            self.sm_vector = self.sm_vector.clone().cuda()
        self.num_bins = num_bins
        self.bins = []
        self.get_equal_bins()

    def get_equal_bins(self):
        for i in range(self.num_bins):
            self.bins.append(torch.tensor(1 / self.num_bins * (i + 1)))

    def get_samples_mask_bins(self):
        mask_list = []
        for i in range(self.num_bins):
            if i == 0:
                mask_list.append(self.sm_vector <= self.bins[i])
            else:
                mask_list.append(
                    (self.bins[i - 1] < self.sm_vector)
                    * (self.sm_vector <= self.bins[i])
                )
        return mask_list


def RBS(data, probs, val_logits, val_labels, test_logits, num_bins):
    def create_adjacency_matrix(graph):
        index_1 = [edge[0] for edge in graph.edges()] + [
            edge[1] for edge in graph.edges()
        ]
        index_2 = [edge[1] for edge in graph.edges()] + [
            edge[0] for edge in graph.edges()
        ]
        values = [1 for edge in index_1]
        node_count = max(max(index_1) + 1, max(index_2) + 1)
        A = sp.sparse.coo_matrix(
            (values, (index_1, index_2)),
            shape=(node_count, node_count),
            dtype=np.float32,
        )
        return A

    graph = to_networkx(data)
    A = create_adjacency_matrix(graph).todense()

    # Calculate agg. probs
    AP = A * probs
    AP = torch.tensor(AP)
    num_neighbors = A.sum(1)
    num_neighbors = torch.tensor(num_neighbors)
    AP[torch.where(num_neighbors == 0)[0]] = 1
    num_neighbors[torch.where(num_neighbors == 0)[0]] = 1
    y_pred = torch.tensor(probs).max(1)[1]
    AP = AP / num_neighbors.expand(AP.shape[0], AP.shape[1])
    conf_AP = []
    for i in range(AP.shape[0]):
        conf_AP.append(AP[i, y_pred[i]])
    sm_prob = np.array(conf_AP)

    # Calculate val and test bins_mask_list
    sm_val = sm_prob[data.val_mask.detach().cpu().numpy()]
    sm_TS_model = bin_mask_eqdis(num_bins, sm_val)
    bins_mask_list = sm_TS_model.get_samples_mask_bins()
    sm_test = sm_prob[data.test_mask.detach().cpu().numpy()]
    sm_TS_model = bin_mask_eqdis(num_bins, sm_test)
    bins_mask_list_test = sm_TS_model.get_samples_mask_bins()

    # Learn temperature
    T_list = []
    for i in range(num_bins):
        TS_model = TemperatureScaling_bins()
        T = TS_model.fit(val_logits[bins_mask_list[i]], val_labels[bins_mask_list[i]])
        T_list.append(torch.tensor(T[1]))

    def get_rescaled_logits(T_list, logits, bins_mask_list):
        T = torch.zeros_like(logits)
        for i in range(num_bins):
            # The i-th bin logits
            logits_i = logits[bins_mask_list[i]]
            # Expand temperature to match the size of logits
            T_i = T_list[i].expand(logits_i.size(0), logits_i.size(1))
            T[bins_mask_list[i], :] = T_i.float()
        logits0 = logits / T
        return logits0

    cal_probs_test = torch.softmax(
        get_rescaled_logits(T_list, torch.tensor(test_logits), bins_mask_list_test), 1
    )

    return cal_probs_test
