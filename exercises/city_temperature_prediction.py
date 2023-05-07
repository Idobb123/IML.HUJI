import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # highest temp recorded in cape town, amsterdam, amman or israel is 49.9C (in israel)
    # lowest temp recorded in those cities was -19C in Amsterdam
    df = df[(df["Temp"] >= -20) & (df["Temp"] <= 50)]

    df["Year"] = df["Year"].astype(str)
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    isr_df = df.loc[df["Country"] == "Israel"]
    Q2_1fig = px.scatter(x=isr_df["DayOfYear"], y=isr_df["Temp"], color=isr_df["Year"])
    Q2_1fig.update_layout(title="Temperature in Israel as a function of the day of year",
                          xaxis_title="Day Of Year",
                          yaxis_title="Temperature")
    Q2_1fig.write_image("Q2.1_figure_israel_daily_temp.png")

    isr_std_month_temp = isr_df.groupby(["Month"])["Temp"].agg("std")
    Q2_2fig = px.bar(isr_std_month_temp, y="Temp", title="Temprature STD over the years for each "
                                                         "month").update_layout(yaxis_title="Temprature STD")
    Q2_2fig.write_image("Q2.2_figure_israel_monthly_std_temp.png")

    # Question 3 - Exploring differences between countries
    mean_and_std_temp = df.groupby(["Month", "Country"])["Temp"].agg(["mean", "std"])
    Q3_fig = px.line(mean_and_std_temp.reset_index(), line_group="Country", x="Month", y="mean", error_y="std",
                     color="Country")
    Q3_fig.update_layout(title="Average Monthly Temperature by country",
                         xaxis_title="Month",
                         yaxis_title="Average Temperature")
    Q3_fig.write_image("Q3_figure_monthly_temp_by_country.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(isr_df["DayOfYear"], isr_df["Temp"])
    k_value_list = np.arange(1, 11)
    loss_list = np.ndarray([10])
    for k in k_value_list:
        PR_model = PolynomialFitting(k).fit(train_X, train_y)
        loss_list[k-1] = np.round(PR_model.loss(test_X, test_y), 2)

    Q4_fig = px.bar(x=k_value_list, y=loss_list)
    Q4_fig.update_layout(title="model error for diffrent k values",
                         xaxis_title="k value",
                         yaxis_title="loss value")
    Q4_fig.write_image("Q4_loss_on_test_israel.png")
    print(loss_list)


    # Question 5 - Evaluating fitted model on different countries
    # choosing k=5 based on Q4 and fitting entire model
    PR_model = PolynomialFitting(5).fit(isr_df["DayOfYear"], isr_df["Temp"])
    country_list = ["The Netherlands", "South Africa", "Jordan"]
    loss_list = np.ndarray([3])
    for i, country in enumerate(country_list):
        c_df = df.loc[df["Country"] == country]
        loss_list[i] = np.round(PR_model.loss(c_df["DayOfYear"], c_df["Temp"]), 2)

    Q5_fig = px.bar(x=country_list, y=loss_list)
    Q5_fig.update_layout(title="Israel Based Model Loss On Other Country's",
                         xaxis_title="Country", yaxis_title="Loss value")
    Q5_fig.write_image("Q5_loss_on_other_country's_based_on_isr_model.png")


