import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from plotly.subplots import make_subplots

import plotly.graph_objects as go

FILE_PREFIX = ""


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def f(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return f, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Q1 : plotting descent path of L1 and L2 for different values of eta
    for module, m_name in [(L2, "L2"), (L1, "L1")]:
        sub_plot_titles = ["\u03B7=1", "\u03B7=0.1", "\u03B7=0.01", "\u03B7=0.001"]
        conv_rate_fig = make_subplots(2, 2, subplot_titles=sub_plot_titles)
        best_loss = np.inf
        best_eta = 0
        for i, eta in enumerate(etas):
            callback, values, weights = get_gd_state_recorder_callback()
            model = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            model.fit(module(init), np.ndarray((0,)), np.ndarray((0,)))
            title = f"for {m_name} with \u03B7={eta}"
            module_dp_fig = plot_descent_path(module=module, descent_path=np.array(weights), title=title)
            module_dp_fig.write_image(FILE_PREFIX + f"{m_name}_Descent_Path_for_eta={eta}.png")
            conv_rate_fig.add_trace(go.Scatter(x=np.array(range(1, 1001)), y=values, showlegend=False),
                                    row=i//2 + 1, col=(i % 2) + 1)
            if values[-1] < best_loss:
                best_loss = values[-1]
                best_eta = eta
        print(f"Best {m_name} loss is achieved with eta={best_eta} and its value is: {np.round(best_loss,5)}")
        conv_rate_fig.update_layout(title=f"Convergence rate of {m_name} module for a selection of \u03B7 values",
                                    xaxis_title="Iterations", yaxis_title=f"{m_name} Norm value", margin=dict(t=100))
        conv_rate_fig.write_image(FILE_PREFIX + f"{m_name}_conv_fig.png")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))).fit(X_train,
                                                                                                        y_train)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test))
    # now selecting the best alpha
    best_alpha_arg = np.argmax(tpr - fpr)
    best_alpha = thresholds[best_alpha_arg]

    # now fitting the model with the best alpha value:
    model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)), alpha=best_alpha)\
        .fit(X_train, y_train)
    test_loss = np.round(model.loss(X_test, y_test), 3)

    q8_fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                                        name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}"),
                             go.Scatter(x=[fpr[best_alpha_arg]], y=[tpr[best_alpha_arg]], text=best_alpha,
                                        mode="markers", marker=dict(size=14, color="blue", symbol="x"),
                                        name="Optimal \u03B1 value")],
        layout=go.Layout(title=f"ROC of Unregularized Logistic Regression Model over test set<br>Optimal \u03B1 value "
                               f"of {np.round(best_alpha,3)} with test error of {test_loss}",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"),
                         margin=dict(t=100)))
    q8_fig.write_image(FILE_PREFIX + "q8_ROC_curve.png")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lambda_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for penalty in ("l1", "l2"):
        model_score_list = np.ndarray((lambda_values.shape[0]))
        for i, lam in enumerate(lambda_values):
            model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                       penalty=penalty, lam=lam)
            train_score, validation_score = cross_validate(model, X_train, y_train, misclassification_error)
            model_score_list[i] = validation_score
        opt_lambda = lambda_values[np.argmin(model_score_list)]

        # now fitting model again and calculating the loss
        opt_model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                       penalty=penalty, lam=opt_lambda).fit(X_train, y_train)
        opt_test_score = opt_model.loss(X_test, y_test)
        print("\n")
        print("optimal lambda value for " + penalty + " model: ", opt_lambda)
        print("optimal test score for " + penalty + " model: ", np.round(opt_test_score, 5))
        print("\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    #compare_exponential_decay_rates()
    fit_logistic_regression()
