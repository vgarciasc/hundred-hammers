from __future__ import annotations
from typing import Tuple
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from .config import hh_logger
from .hyperparameters import find_hyperparam_grid, find_hyperparam_random
from .metric_alias import process_metric


class HyperOptimizer(ABC):
    """
    Hyperparameter Optmizer interface.

    :param metric: function that calculates the error of the predicitons of a model compared with the real dataset.
    :param metric_params: parameters of the metric function.
    """

    def __init__(self, metric: str | callable | Tuple[str, callable, dict] = "MSE"):
        if isinstance(metric, tuple | list):
            _, metric_fn, metric_params = metric
        else:
            _, metric_fn, metric_params = process_metric(metric)
        self.metric_fn = make_scorer(metric_fn, **metric_params)

    def set_metric(self, metric: callable):
        """
        Replaces the evaluation metric.
        """

        _, metric, metric_params = process_metric(metric)
        self.metric_fn = make_scorer(lambda y_true, y_pred: metric(y_true, y_pred, **metric_params))

    @abstractmethod
    def best_params(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator, param_grid: dict = None):
        """
        Obtains the best set parameters for the given model and dataset.
        """


class HyperOptimizerGridSearch(HyperOptimizer):
    """
    Grid Search Hyperparameter Optimizer.

    :param metric: function that calculates the error of the predicitons of a model compared with the real dataset.
    :param metric_params: parameters of the metric function.
    :param n_folds_tune: number of splits in cross validation for grid search.
    :param n_grid_points: amount of points to choose per parameter when the grid is constructed.
    """

    def __init__(self, metric: str | callable = "MSE", n_folds_tune: int = 5, n_grid_points: int = 10):
        super().__init__(metric)
        self.n_folds_tune = n_folds_tune
        self.n_grid_points = n_grid_points

    def best_params(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator, param_grid: dict = None):
        if not param_grid:
            hh_logger.info(f"No specified hyperparameter grid for {type(model).__name__}. Generating hyperparameter grid.")
            param_grid = find_hyperparam_grid(model, self.n_grid_points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search_model = GridSearchCV(model, param_grid, scoring=self.metric_fn, n_jobs=-1, cv=self.n_folds_tune)
            grid_search_model.fit(X, y)

        results = pd.DataFrame(grid_search_model.cv_results_).dropna()
        best_params_df = results[results["rank_test_score"] == results["rank_test_score"].min()]
        best_params = best_params_df.head(1)["params"].values[0]

        return best_params


class HyperOptimizerRandomSearch(HyperOptimizer):
    """
    Grid Search Hyperparameter Optimizer.

    :param metric: function that calculates the error of the predicitons of a model compared with the real dataset.
    :param metric_params: parameters of the metric function.
    :param n_folds_tune: number of splits in cross validation for grid search.
    :param n_grid_points: amount of points to choose per parameter when the grid is constructed.
    """

    def __init__(self, metric: str | callable = "MSE", n_folds_tune: int = 5, n_iter: int = 10):
        super().__init__(metric)
        self.n_folds_tune = n_folds_tune
        self.n_iter = n_iter

    def best_params(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator, param_grid: dict = None):
        if not param_grid:
            hh_logger.info(f"No specified hyperparameter grid for {type(model).__name__}. Generating hyperparameter grid.")
            param_grid = find_hyperparam_random(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search_model = RandomizedSearchCV(model, param_grid, scoring=self.metric_fn, n_jobs=-1, cv=self.n_folds_tune, n_iter=self.n_iter)
            grid_search_model.fit(X, y)

        results = pd.DataFrame(grid_search_model.cv_results_).dropna()
        best_params_df = results[results["rank_test_score"] == results["rank_test_score"].min()]
        best_params = best_params_df.head(1)["params"].values[0]

        return best_params
