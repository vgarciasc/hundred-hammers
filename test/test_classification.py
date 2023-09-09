# pylint: skip-file
import pytest
import logging
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from hundred_hammers import hh_logger
from hundred_hammers import HundredHammersClassifier
from hundred_hammers.plots import plot_batch_results, plot_confusion_matrix
from hundred_hammers.model_zoo import KNeighborsClassifier

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

def test_complete_optim_hyperparams():
    data = load_iris()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersClassifier()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, n_grid_points=4)
