from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = (1 / len(X)) * np.sum(X)
        # as seen in the course book for the unbiased version we need to divide
        # by the number of  degrees of freedom. particularly here it is the
        # number of  samples minus the 1 parameter (the expectation) that we
        # used. for the biased version we simply divide by the number of
        # samples
        if self.biased_:
            self.var_ = (1 / len(X)) * np.sum(np.power(X - self.mu_, 2))
        else:
            self.var_ = (1 / (len(X) - 1)) * np.sum(np.power(X - self.mu_, 2))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        # for each sample x we will calculate the pdf using the formula seen
        # in class: so 1/sqrt(2*pi*var) * exp((-1/(2*var))*(x-mu)**2)
        return np.exp((-1 / (2 * self.var_)) * np.power(X - self.mu_, 2)) / \
               np.sqrt(2 * np.pi * self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # we saw that the likelihood for a gaussian model is of the form:
        #           exp(-Sum((x-mu)^2)/2*var)/(2*pi*var)^m/2
        # so we can take the log of the expression will will and using log
        # rules get to the following calculation:

        return (-len(X)/2) * np.log(2*np.pi*sigma) + (-np.sum((X-mu)**2))/(
                2*sigma)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X, axis=0)
        # now for the covariance matrix according to the formula in the book
        # under definition 1.2.7
        c_X = X - self.mu_  # centering the matrix
        self.cov_ = (c_X.T @ c_X) / (len(X) - 1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        # calculating PDF according to the fit of the current model
        denominator = np.sqrt(((2 * np.pi)**len(self.mu_)) *
                                   np.linalg.det(self.cov_))
        inv_cov = np.linalg.inv(self.cov_)  # the inverted covariance matrix
        c_X = X - self.mu_                  # the centered sample array

        # im assuming there's a better way to do this but i couldn't think of it
        arg_array = []
        for i in range(len(X)):
            cur_sample = c_X[i:i+1]
            arg_array.append(np.sum((-(cur_sample @ inv_cov @
                                       cur_sample.T)/2)))
        return np.exp(arg_array) / denominator


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # we shall calculate the log_likelihood according the the formula
        # derived in Q13 using the einsum function that someone luckly asked
        # about in the forums!

        p1 = (len(X) * np.log((2 * np.pi)**len(mu))) / 2
        p2 = (len(X) * np.log(np.linalg.det(cov))) / 2

        c_X = X - mu
        inv_cov = np.linalg.inv(cov)
        # einsum solution is also possible
        # p3 = np.einsum("bi,ij,bj", c_X, inv_cov, c_X) / 2
        first_matrix = c_X @ inv_cov
        p3 = np.sum(first_matrix * c_X) / 2
        return -p1 - p2 - p3




