from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    true_mu = 10
    true_var = 1
    sample_size = 1000

    # Question 1 - Draw samples and print fitted model
    # draws 1000 samples from N(10,1)
    X = np.random.normal(true_mu, true_var, (sample_size,))
    UV = UnivariateGaussian()
    UV.fit(X)
    print("Estimated values of expectation and variance rounded to 3 decimal "
          "places: (" + str(np.round(UV.mu_, 3)) + ", " +
          str(np.round(UV.var_, 3)) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    model_dist_arr = []
    sample_size_arr = []
    for i in range(10, sample_size + 10, 10):
        UV.fit(X[:i])  # fitting the model until the i'th sample
        model_dist_arr.append(abs(true_mu - UV.mu_))
        sample_size_arr.append(i)

    # now to plot the graph
    q2_fig = go.Figure(go.Scatter(x=sample_size_arr, y=model_dist_arr))
    q2_fig.update_layout(title="value of error as a function of "
                              "sample size",
    xaxis_title="Sample size",
    yaxis_title="value of error: |μ_hat - μ|",
                        font=dict(size=18))
    q2_fig.write_image("q2_fig.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    q3_fig = go.Figure(go.Scatter(x=X, y=UV.pdf(X), mode="markers"))
    q3_fig.update_layout(title="PDF of a N(10,1) as a function of  sample "
                               "value using fitted model",
                         xaxis_title="N(10,1) Sample value",
                         yaxis_title="PDF of draw sample",
                         font=dict(size=14))
    q3_fig.write_image("q3_fig.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    true_mu = np.array([0, 0, 4, 0])
    true_cov = np.array([[1, 0.2, 0, 0.5],
                           [0.2, 2, 0, 0],
                           [0, 0, 1, 0],
                           [0.5, 0, 0, 1]])
    sample_size = 1000
    X = np.random.multivariate_normal(true_mu, true_cov, size=sample_size)
    MV = MultivariateGaussian()
    MV.fit(X)
    print("value of estimated expectation rounded to 3 decimal placed: " + str(
        np.round(MV.mu_, 3)))
    print("value of estimated covariance rounded to 3 decimal places:\n" + str(
        np.round(MV.cov_, 3)))

    # Question 5 - Likelihood evaluation
    # NOTE: this runs very slowly, there's probably a better way to do this
    lin_sample_size = 200
    ll_val_array = np.zeros((lin_sample_size, lin_sample_size))
    f_arr = np.linspace(-10, 10, lin_sample_size)
    for i in range(lin_sample_size):
        for j in range(lin_sample_size):
            cur_values = np.array([f_arr[i], 0, f_arr[j], 0])
            ll_val_array[i, j] = MV.log_likelihood(cur_values, true_cov, X)

    q5_fig = go.Figure(go.Heatmap(x=f_arr, y=f_arr, z=ll_val_array))
    q5_fig.update_layout(title="log-likelihood of multivariate normal "
                               "distribution as a function of the "
                               "expectation of features 1 and 3",
                         yaxis_title="expectation value of feature 1",
                         xaxis_title="expectation value of feature 3",
                         template="simple_white",
                         font=dict(size=10))
    q5_fig.write_image("q5_fig.png")

    # Question 6 - Maximum likelihood
    flat_ind = np.argmax(ll_val_array)  # finding the ll with the highest value
    # now using unravel to find what is the right conversion to a (200,+
    # 200) array
    right_ind = np.unravel_index(flat_ind, ll_val_array.shape)
    f1_val = np.round(f_arr[right_ind[0]], 3)
    f3_val = np.round(f_arr[right_ind[1]], 3)
    print("value's of f1 and f3 that achieved maximum log-likelihood "
          "rounded to 3 decimal places: " +
          "\nf1 = " + str(f1_val) + "\nf3 = " + str(f3_val))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
