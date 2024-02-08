# pylint: skip-file
import pandas as pd
import pytest
import logging
import time
import os
from sklearn.datasets import make_classification
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


@pytest.mark.skip(reason="Test for progress bars")
def test_progress_nohyper():
    X, y = make_classification(n_samples=1000, n_features=10)

    hh_models = HundredHammersClassifier(seed_strategy='sequential', show_progress_bar=True)

    df_results_1 = hh_models.evaluate(X, y, optim_hyper=False)

@pytest.mark.skip(reason="Test for progress bars")
def test_progress_hyper():
    X, y = make_classification(n_samples=200, n_features=10)

    hh_models = HundredHammersClassifier(seed_strategy='sequential', show_progress_bar=True)

    df_results_1 = hh_models.evaluate(X, y, optim_hyper=True)


if __name__ == "__main__":
    hh_logger.setLevel("ERROR")
    os.system('clear' if os.name == 'posix' else 'cls')

    test_progress_nohyper()
    input("Next test.")
    os.system('clear' if os.name == 'posix' else 'cls')
    
    test_progress_hyper()
    input("Next test.")
    os.system('clear' if os.name == 'posix' else 'cls')
