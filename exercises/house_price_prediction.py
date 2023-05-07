from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

AVG_PRICE = 0
AVG_BEDROOMS = 0
AVG_BATHROOMS = 0
AVG_SQFT_LIVING = 0
AVG_SQFT_LOT = 0
AVG_FLOORS = 0
AVG_WATERFRONT = 0
AVG_VIEW = 0
AVG_CONDITION = 0
AVG_GRADE = 0
AVG_SQFT_ABOVE = 0
AVG_SQFT_BASEMENT = 0
AVG_YEAR_BUILT = 0
AVG_DECADE_BUILT = 0
AVG_RENOV_STAT = 0
AVG_SQFT_LIVING15 = 0
AVG_SQFT_LOT15 = 0


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # if y is not None than we are processing the train data
    if y is not None:
        X = preprocess_train_design_matrix(X)
        y = y.loc[X.index]
        y = y.dropna()
        y = y.loc[y['price'] >= 0]  # removes all of the negative price values
        X = X.loc[y.index]
        calculate_mean_values(X, y)
        return X, y
    else:  # processing the test data
        X = preprocess_test_design_matrix(X)
        X = restructure_matrix_data(X)
        return X


def calculate_mean_values(X: pd.DataFrame, y: pd.Series):
    """
    gets the mean values of the relevent columns. used later when preprocesing the test
    Parameters : same as the process data function
    ---------
    Returns None
    -------
    """
    global AVG_PRICE
    global AVG_BEDROOMS
    global AVG_BATHROOMS
    global AVG_SQFT_LIVING
    global AVG_SQFT_LOT
    global AVG_FLOORS
    global AVG_VIEW
    global AVG_CONDITION
    global AVG_GRADE
    global AVG_SQFT_ABOVE
    global AVG_SQFT_BASEMENT
    global AVG_SQFT_LIVING15
    global AVG_SQFT_LOT15

    AVG_PRICE = y["price"].mean()
    AVG_BEDROOMS = X["bedrooms"].mean()
    AVG_BATHROOMS = X["bathrooms"].mean()
    AVG_SQFT_LIVING = X["sqft_living"].mean()
    AVG_SQFT_LOT = X["sqft_lot"].mean()
    AVG_FLOORS = X["floors"].mean()
    AVG_VIEW = X["view"].mean()
    AVG_CONDITION = X["condition"].mean()
    AVG_GRADE = X["grade"].mean()
    AVG_SQFT_ABOVE = X["sqft_above"].mean()
    AVG_SQFT_BASEMENT = X["sqft_basement"].mean()
    AVG_SQFT_LIVING15 = X["sqft_living15"].mean()
    AVG_SQFT_LOT15 = X["sqft_lot15"].mean()
    return


def preprocess_train_design_matrix(X: pd.DataFrame):
    """
    preprocesses the train design matrix
    Parameters
    ----------
    X the train design matrix

    Returns
    -------
    the processed design matrix
    """

    X = clean_train_matrix_data(X)
    X = restructure_matrix_data(X)
    return X


def clean_train_matrix_data(X: pd.DataFrame):
    """
    cleans the data of the train or test set
    Parameters
    ----------
    X - train set

    Returns
    -------
    cleaned train set
    """
    X = X.dropna()
    X = X.drop_duplicates()
    X = X.drop(['id', 'date', "lat", "long"], axis=1)

    # 1. now to drop rows that have bad data with respect to numerical values that should be positive
    for column in ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_above",
                   "yr_built", "sqft_living15", "sqft_lot15"]:
        X = X.drop(X[X[column] <= 0].index)

    global AVG_YEAR_BUILT
    AVG_YEAR_BUILT = X['yr_built'].mean()

    # 2. now for the columns that can also be zero
    for column in ["sqft_basement", "yr_renovated"]:
        X = X.drop(X[X[column] < 0].index)

    # 3. and lastly columns that have a certain range of values (with respect the the table on the websibe)
    X = X.loc[(X["waterfront"] >= 0) & (X["waterfront"] <= 1)]
    X = X.loc[(X["view"] >= 0) & (X["view"] <= 4)]
    X = X.loc[(X["condition"] >= 1) & (X["view"] <= 5)]
    X = X.loc[(X["grade"] >= 1) & (X["grade"] <= 13)]

    return X


def restructure_matrix_data(X: pd.DataFrame):
    """
    restructures the design matrix. creates some new features using old ones and some dummy variables
    Parameters
    ----------
    X - the train design matrix

    Returns
    -------
    the design matrix after some restructuring
    """

    # 4. turning year renovated into a status variabe where we have 0 representing not renovated, 1 for renovated a
    # long time ago and 2 representing a recent renovation. this is in a sense natural ordering of things since
    # the later the renovation occured the more valueable?

    X["yr_renovated"] = X["yr_renovated"].astype(int)
    X.loc[(X["yr_renovated"] > 0) & (X["yr_renovated"] < 2000), "yr_renovated"] = 1
    X.loc[(X["yr_renovated"] >= 2000), "yr_renovated"] = 2
    X.rename(columns={"yr_renovated": "renovation_status"}, inplace=True)

    # 5. now turning year_built into newly_built so we can parse it more easily

    X["yr_built"] = X["yr_built"].astype(int)
    X.loc[(X["yr_built"] < np.percentile(X["yr_built"], 70)), "yr_built"] = 0
    X.loc[(X["yr_built"] > 0), "yr_built"] = 1
    X.rename(columns={"yr_built": "newly_built"}, inplace=True)

    # 6. turning the zipcode into a categorical variable

    X["zipcode"] = X["zipcode"].astype(int)
    X = pd.get_dummies(X, prefix="zipcode_", columns=["zipcode"])
    return X


def preprocess_test_design_matrix(X: pd.DataFrame):
    """
    processes the test design matrix. notably this function does not delete any lines
    this function will replace missing values with the mean values from the train sample
    Parameters
    ----------
    X - design matrix of the test set

    Returns
    -------
    the design matrix of the test set post-processing without deleting any samples (rows)
    """
    X = X.drop(['id', 'date', "lat", "long"], axis=1)

    feature_mean_list = [AVG_BEDROOMS, AVG_BATHROOMS, AVG_SQFT_LIVING, AVG_SQFT_LOT, AVG_FLOORS, AVG_WATERFRONT, AVG_VIEW,
                 AVG_CONDITION, AVG_GRADE, AVG_SQFT_ABOVE, AVG_SQFT_BASEMENT, AVG_YEAR_BUILT, AVG_RENOV_STAT,
                 AVG_SQFT_LIVING15, AVG_SQFT_LOT15]
    feature_list = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition",
                    "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15",
                    "sqft_lot15"]
    for i in range(len(feature_mean_list)):
        cur_feature = feature_list[i]
        cur_mean = feature_mean_list[i]
        X.loc[(X[cur_feature] < 0) | (X[cur_feature].isna()), cur_feature] = cur_mean

    X.loc[X["zipcode"].isna(), "zipcode"] = 0

    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for feature in X:
        #TODO -> for the marker of this ex: please look at Answers pdf for this question
        if "zipcode_" in feature:
            continue
        cur_cov_matrix = np.cov(X[feature], y['price'])  # this is a 2x2 matrix and we need one of the values not on the
        # diag (they are both the same since its symmetrical)

        p_cor = cur_cov_matrix[0, 1] / (np.std(X[feature]) * np.std(y))
        fig = go.Figure(go.Scatter(x=X[feature], y=y['price'], mode="markers"))
        fig.update_layout(title=str(feature) + " values' correlation with the response" + " <br>at pearson "
                                                                                           "correlation "
                                                                                           "value of " + str(p_cor[0]),
                          xaxis_title=str(feature), yaxis_title="response", font=dict(size=14))

        fig.write_image(output_path + "/pearson.correlation." + str(feature) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.loc[:, df.columns != "price"], df.loc[:, ["price"]])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)  # processing the train design and response
    test_y = test_y.dropna()  # dropping Nan values since we cant use these anyways
    test_X = test_X.loc[test_y.index]  # dropping the corresponding rows from test_X
    test_X = preprocess_data(test_X)  # processing the test design matrix
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)


    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    p_list = np.arange(10, 101)
    mean_list = np.ndarray([91])
    var_list = np.ndarray([91])
    cur_p_list = np.ndarray([10])
    LR = LinearRegression(include_intercept=True)
    for p in p_list:
        for i in range(10):
            partial_X = train_X.sample(frac=(p / 100))
            partial_response = train_y.loc[partial_X.index]
            cur_p_list[i] = LR.fit(partial_X, partial_response).loss(test_X.to_numpy(), test_y)

        mean_list[p - 10] = np.mean(cur_p_list)
        var_list[p - 10] = 2 * np.std(cur_p_list)

    fig = go.Figure([go.Scatter(x=p_list, y=mean_list, mode="markers+lines",
                                name="Mean Prediction"),
                     go.Scatter(x=p_list,
                                y=mean_list - var_list,
                                fill=None, mode="lines",
                                line=dict(color="lightgrey")),
                     go.Scatter(x=p_list,
                                y=mean_list + var_list,
                                fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"),
                                showlegend=False)])
    fig.update_layout(title="Expected loss as a function of the percentage of the test set taken <br>with confidence interval of 2 " \
            "standard deviations", xaxis_title="p of training set", yaxis_title="loss value over test set")
    fig.write_image("q4_ex2_fig.png")








