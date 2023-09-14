# pylint: skip-file

import pandas as pd
import logging
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from hundred_hammers import hh_logger
from hundred_hammers import HundredHammersRegressor
from hundred_hammers.plots import plot_batch_results, plot_regression_pred
from hundred_hammers.model_zoo import (
    DecisionTreeRegressor,
    LinearRegression,
    DummyRegressor
)

import warnings
from sklearn.exceptions import ConvergenceWarning

hh_logger.setLevel(logging.DEBUG)

def test_complete_default():
    data = load_diabetes()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersRegressor()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=False)
    assert type(df_results) is pd.DataFrame

def test_complete_optim_hyperparams():
    data = load_diabetes()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersRegressor()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, n_grid_points=4)
    assert type(df_results) is pd.DataFrame
