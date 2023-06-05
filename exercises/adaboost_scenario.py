import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    AD_model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    iter_list = np.array(range(1, n_learners + 1))
    test_pred_list = []
    train_pred_list = []
    for i in iter_list:
        train_pred_list.append(AD_model.partial_loss(train_X, train_y, i))
        test_pred_list.append(AD_model.partial_loss(test_X, test_y, i))
    Q1_fig = go.Figure([go.Scatter(x=iter_list, y=train_pred_list, name="train error"),
                       go.Scatter(x=iter_list, y=test_pred_list, name="test error")])
    Q1_fig.update_layout(title=f"Train and Test error of Adaboost learner as a function of number of learners<br>       "
                               f"noise value of {noise}",
                         xaxis_title="number of learners",
                         yaxis_title="error")
    Q1_fig.write_image(f"Q1_fig_with_noise_{noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    Q2_fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{T[i]} Iterations" for i in range(4)],
                           horizontal_spacing=0.12, vertical_spacing=0.12)
    symbols = np.where(test_y >= 0, "diamond", "cross")
    for i in range(4):
        partial_pred_func = lambda X: AD_model.partial_predict(X, T[i])
        Q2_fig.add_traces([decision_surface(partial_pred_func, lims[0], lims[1], showscale=False),
                          go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                     marker=dict(color=test_y, symbol=symbols, colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))],
                         rows=(i//2) + 1, cols=(i % 2) + 1)
    Q2_fig.update_layout(title="Decision Boundary Of AdaBoost Ensemble As a Function Of Number Of Iterations",
                         margin=dict(t=100), title_font=dict(size=16))
    Q2_fig.write_image("Q2_fig.png")

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
