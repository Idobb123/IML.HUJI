from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # create an index list for splitting the array later
    ind_list = np.arange(len(X))
    split_ind_list = np.split(ind_list, cv)

    # calculate the error for each fold
    train_score, val_score = 0, 0
    for i in range(cv):
        train_indices = np.delete(ind_list, split_ind_list[i])
        # TODO -> maybe need to copy the estimator
        cur_est = estimator.fit(X[train_indices], y[train_indices])
        train_score += scoring(y[train_indices], cur_est.predict(X[train_indices]))
        val_score += scoring(y[split_ind_list[i]], cur_est.predict(X[split_ind_list[i]]))

    return train_score / cv, val_score / cv




