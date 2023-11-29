# pylint: skip-file

import pandas as pd
import logging
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer, RobustScaler, KBinsDiscretizer

from hundred_hammers import hh_logger
from hundred_hammers import HundredHammersRegressor, HyperOptimizerGridSearch, HyperOptimizerRandomSearch
from hundred_hammers.plots import plot_batch_results, plot_regression_pred
from hundred_hammers.model_zoo import (
    DecisionTreeRegressor,
    LinearRegression,
    LinearSVR,
    DummyRegressor
)

import warnings
from sklearn.exceptions import ConvergenceWarning

# hh_logger.setLevel(logging.DEBUG)
hh_logger.setLevel(logging.CRITICAL)

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
    
    # Small amount of models to reduce running time
    models = [
        ('Decision Tree', DecisionTreeRegressor(), None),
        ('Linear Regression', LinearRegression(), None),
        ('SVM', LinearSVR(), None),
        ('Dummy', DummyRegressor(), None)
    ]

    # Create the model
    hh = HundredHammersRegressor(models=models)

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True)
    assert type(df_results) is pd.DataFrame

def test_complete_optim_hyperparams_gridsearch():
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # Small amount of models to reduce running time
    models = [
        ('Decision Tree', DecisionTreeRegressor(), None),
        ('Linear Regression', LinearRegression(), None),
        ('SVM', LinearSVR(), None),
        ('Dummy', DummyRegressor(), None)
    ]
    
    # Create the model
    hh = HundredHammersRegressor(models=models)

    # Create hyperparameter optimizer
    hoptim = HyperOptimizerGridSearch("MSE")

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, hyperoptimizer=hoptim)
    assert type(df_results) is pd.DataFrame

def test_complete_optim_hyperparams_randomsearch():
    data = load_diabetes()
    X = data.data
    y = data.target
    
    # Small amount of models to reduce running time
    models = [
        ('Decision Tree', DecisionTreeRegressor(), None),
        ('Linear Regression', LinearRegression(), None),
        ('SVM', LinearSVR(), None),
        ('Dummy', DummyRegressor(), None)
    ]

    # Create the model
    hh = HundredHammersRegressor(models=models)

    # Create hyperparameter optimizer
    hoptim = HyperOptimizerRandomSearch()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, hyperoptimizer=hoptim)
    assert type(df_results) is pd.DataFrame

def test_cross_validations():
    data = load_diabetes()
    X = data.data
    y = data.target

    # Small amount of models to reduce running time
    models = [
        ('Decision Tree', DecisionTreeRegressor(), None),
        ('Linear Regression', LinearRegression(), None),
        ('SVM', LinearSVR(), None),
        ('Dummy', DummyRegressor(), None)
    ]

    hh = HundredHammersRegressor(models=models, cross_validator=LeaveOneOut, cross_validator_params={})
    df_results_1 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, cross_validator=KFold, cross_validator_params={'n_splits': 5})
    df_results_2 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, cross_validator=KFold, cross_validator_params={'n_splits': 5})
    df_results_3 = hh.evaluate(X, y, optim_hyper=False)

    assert df_results_2.equals(df_results_3)
    assert not df_results_1.equals(df_results_3)

def test_normalization():
    data = load_diabetes()
    X = data.data
    y = data.target

    # Small amount of models to reduce running time
    models = [
        ('Decision Tree', DecisionTreeRegressor(), None),
        ('Linear Regression', LinearRegression(), None),
        ('SVM', LinearSVR(), None),
        ('Dummy', DummyRegressor(), None)
    ]

    hh = HundredHammersRegressor(models=models, input_transform=MinMaxScaler)
    df_results_1 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, input_transform=MinMaxScaler())
    df_results_2 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, input_transform="MinMax")
    df_results_3 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, input_transform="Standard")
    df_results_4 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersRegressor(models=models, input_transform="MaxAbs")
    df_results_5 = hh.evaluate(X, y, optim_hyper=False)
