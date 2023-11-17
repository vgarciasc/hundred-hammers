# pylint: skip-file
import pandas as pd
import pytest
import logging
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer, RobustScaler, KBinsDiscretizer

from hundred_hammers import hh_logger
from hundred_hammers import HundredHammersClassifier
from hundred_hammers.plots import plot_batch_results, plot_confusion_matrix
from hundred_hammers.model_zoo import (
    LogisticRegression,
    DecisionTreeClassifier,
    KNeighborsClassifier,
    DummyClassifier
)

import warnings
from sklearn.exceptions import ConvergenceWarning

hh_logger.setLevel(logging.DEBUG)

def test_complete_default():
    data = load_iris()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersClassifier()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=False)
    assert type(df_results) is pd.DataFrame

def test_complete_optim_hyperparams():
    data = load_iris()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersClassifier()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, n_grid_points=4)
    assert type(df_results) is pd.DataFrame

def test_seed_strategy():
    data = load_iris()
    X = data.data
    y = data.target

    # Small amount of models to reduce running time
    models = [
        ('Logistic Regression', LogisticRegression(), None),
        ('Decision Tree', DecisionTreeClassifier(), None),
        ('KNN', KNeighborsClassifier(), None),
        ('Dummy', DummyClassifier(), None),
    ]

    df_results_1 = HundredHammersClassifier(models=models, seed_cv_strategy='sequential').evaluate(X, y, optim_hyper=False)
    df_results_2 = HundredHammersClassifier(models=models, seed_cv_strategy='sequential').evaluate(X, y, optim_hyper=False)

    assert df_results_1.equals(df_results_2)

    df_results_3 = HundredHammersClassifier(models=models, seed_cv_strategy='random').evaluate(X, y, optim_hyper=False)
    df_results_4 = HundredHammersClassifier(models=models, seed_cv_strategy='random').evaluate(X, y, optim_hyper=False)

    assert not df_results_3.equals(df_results_4)

def test_cross_validations():
    data = load_iris()
    X = data.data
    y = data.target

    # Small amount of models to reduce running time
    models = [
        ('Logistic Regression', LogisticRegression(), None),
        ('Decision Tree', DecisionTreeClassifier(), None),
        ('KNN', KNeighborsClassifier(), None),
        ('Dummy', DummyClassifier(), None),
    ]

    hh = HundredHammersClassifier(models=models, cross_validator=KFold, cross_validator_params={'n_splits': 5})
    df_results_1 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersClassifier(models=models, cross_validator=KFold, cross_validator_params={'n_splits': 5})
    df_results_2 = hh.evaluate(X, y, optim_hyper=False)

    assert df_results_1.equals(df_results_2)

    hh = HundredHammersClassifier(models=models, cross_validator=StratifiedKFold, cross_validator_params={'n_splits': 5})
    df_results_3 = hh.evaluate(X, y, optim_hyper=False)

    hh = HundredHammersClassifier(models=models, cross_validator=LeaveOneOut, cross_validator_params={})
    df_results_4 = hh.evaluate(X, y, optim_hyper=False)

    assert not df_results_3.equals(df_results_4)
