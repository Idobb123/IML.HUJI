from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    # generate the samples
    train_X = X.sample(n=n_samples)
    train_y = y.loc[train_X.index]
    test_X = X.drop(train_X.index)
    test_y = y.loc[test_X.index]
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    r_values, l_values, = np.linspace(0.01, 0.2, n_evaluations), np.linspace(0.01, 1, n_evaluations)
    r_score, l_score = np.empty((n_evaluations, 2)), np.empty((n_evaluations, 2))
    # now calculate the score for different values of lambda
    for i in range(n_evaluations):
        r_score[i] = cross_validate(RidgeRegression(r_values[i]), train_X, train_y, mean_square_error)
        l_score[i] = cross_validate(Lasso(alpha=l_values[i], max_iter=10000), train_X, train_y, mean_square_error)

    Q7_fig = make_subplots(1, 2, subplot_titles=["Ridge Regression", "Lasso Regression"])
    Q7_fig.add_trace(go.Scatter(x=r_values, y=r_score[:, 0], name="Ridge Train Score"), row=1, col=1)
    Q7_fig.add_trace(go.Scatter(x=r_values, y=r_score[:, 1], name="Ridge Validation Score"), row=1, col=1)
    Q7_fig.add_trace(go.Scatter(x=l_values, y=l_score[:, 0], name="Lasso Train Score"), row=1, col=2)
    Q7_fig.add_trace(go.Scatter(x=l_values, y=l_score[:, 1], name="Lasso Validation Score"), row=1, col=2)

    Q7_fig.update_layout(title="Ridge And Lasso Regression models MSE Over Train and Validation Set<br>As a Function "
                               "of Regularization Parameter Value", margin=dict(t=120),
                         xaxis_title="Regularization Parameter Value",
                         yaxis_title="MSE")
    Q7_fig.write_image("Q7_fig.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # finding out which lambda value achieved the best outcome:
    best_r_value = r_values[np.argmin(r_score[:, 1])]
    best_l_value = l_values[np.argmin(l_score[:, 1])]

    #calculating the MSE for the optimal values
    opt_ridge_MSE = RidgeRegression(lam=best_r_value).fit(train_X, train_y).loss(test_X, test_y)
    lasso_y_hat = Lasso(alpha=best_l_value).fit(train_X, train_y).predict(test_X)
    opt_lasso_MSE = mean_square_error(test_y, lasso_y_hat)
    LS_MSE = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)

    #printing the results
    print("Best Ridge value: ", np.round(best_r_value, 5))
    print("Best Lasso value: ", np.round(best_l_value, 5))
    print("Optimal Ridge MSE: ", np.round(opt_ridge_MSE))
    print("Optimal Lasso MSE: ", np.round(opt_lasso_MSE))
    print("LS MSE: ", np.round(LS_MSE))



if __name__ == '__main__':
    np.random.seed(0)
    #raise NotImplementedError()
    select_regularization_parameter()
