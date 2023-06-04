from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

MSE = 1
THR = 0


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        thr_list_plus = np.ndarray((X.shape[1], 2))
        thr_list_minus = np.ndarray((X.shape[1], 2))
        for j in range(X.shape[1]):
            thr_list_plus[j, THR], thr_list_plus[j, MSE] = self._find_threshold(X[:, j], y, 1)
            thr_list_minus[j, THR], thr_list_minus[j, MSE] = self._find_threshold(X[:, j], y, -1)

        if np.min(thr_list_plus[:, MSE]) <= np.min(thr_list_minus[:, MSE]):
            self.j_: int = int(np.argmin(thr_list_plus[:, MSE]))
            self.threshold_, self.sign_ = thr_list_plus[self.j_, THR], 1
        else:
            self.j_ = int(np.argmin(thr_list_minus[:, MSE]))
            self.threshold_, self.sign_ = thr_list_minus[self.j_, THR], -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        #sort the values so we can use cumsum to calculate the loss
        sorted_indexes = np.argsort(values)
        values, labels = values[sorted_indexes], labels[sorted_indexes]

        weights = np.where(np.sign(labels) == sign, np.abs(labels), 0)
        left_losses = np.concatenate([[0], np.cumsum(weights)])
        right_losses = np.concatenate([np.cumsum((np.abs(labels) - weights)[::-1])[::-1], [0]])
        losses = left_losses + right_losses

        min_loss_ind = np.argmin(losses)
        if min_loss_ind == len(values):
            threshold = values[min_loss_ind - 1] + 0.1
        else:
            threshold = values[min_loss_ind]
        return threshold, losses[min_loss_ind] / len(values)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))



