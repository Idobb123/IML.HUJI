import numpy as np
import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_func(model: Perceptron, _, __):
            losses.append(model.loss(X, y))

        model = Perceptron(callback=callback_func)
        model.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(go.Scatter(x=np.arange(0, len(losses)), y=losses, mode="lines"))
        fig.update_layout(title=f"Preceptron Misclassification Error Over {n} <br>Data as a Function Of Iterations",
                          xaxis_title="Number Of Iterations",
                          yaxis_title="Misclassification Error Value")
        fig.write_image(f"Q1_fig_perceptron_loss_over_{n}_data.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in [("gaussian1", "gaussian1.npy"), ("gaussian2", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        LDA_model = LDA().fit(X, y)
        GNB_model = GaussianNaiveBayes().fit(X, y)
        LDA_pred = LDA_model.predict(X)
        GNB_pred = GNB_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Gaussian Naive Bayes predictions with accuracy of "
                                                            f"{round(accuracy(y, GNB_pred),3)} ",
                                                            f"LDA predictions with accuracy of "
                                                            f"{round(accuracy(y, LDA_pred),3)} "))
        fig.update_layout(title_text="Gaussian Naive Bayes and LDA predictions over " + n + " dataset",
                          showlegend=False, width=1000, height=500, title_font=dict(size=24))
        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=GNB_pred, symbol=class_symbols[y])), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=LDA_pred, symbol=class_symbols[y])), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=GNB_model.mu_[:, 0], y=GNB_model.mu_[:, 1], mode="markers",
                                 marker=dict(color="black", symbol="x", size=14)), row=1, col=1)
        fig.add_trace(go.Scatter(x=LDA_model.mu_[:, 0], y=LDA_model.mu_[:, 1], mode="markers",
                                 marker=dict(color="black", symbol="x", size=14)), row=1, col=2)


        # Add ellipses depicting the covariances of the fitted Gaussians g
        for i in range(len(LDA_model.classes_)):
            fig.add_trace(get_ellipse(GNB_model.mu_[i], np.diag(GNB_model.vars_[i])), row=1, col=1)
            fig.add_trace(get_ellipse(GNB_model.mu_[i], LDA_model.cov_), row=1, col=2)
        fig.write_image(f"Q3_fig_over {n}.png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
