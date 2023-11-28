"""
This module provides some alternative names for metrics
implemented in sklearn.
"""

from __future__ import annotations
from typing import Tuple
from sklearn.metrics import get_scorer

metric_alias = {
    "ACC": "accuracy",
    "BACC": "balanced_accuracy",
    "PREC": "precision",
    "PRECW": "precision_weighted",
    "REC": "recall",
    "F1": "f1",
    "F1W": "f1_weighted",
    "ROC": "roc_auc",
    "LogLoss": "neg_log_loss",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}


def process_metric(metric: str | callable, metric_params: dict = None) -> Tuple[str, callable, dict]:
    """
    Converts a metric into a tuple with the name, function call and its parameters

    :param metric: a string or callable that represents the error function
    """

    if isinstance(metric, str):
        # Metric given by its name
        metric_fn_name = metric
        if metric in metric_alias:
            metric_fn_name = metric_alias[metric]

        scorer = get_scorer(metric_fn_name)

        name = metric
        metric_fn = scorer._score_func
        metric_params = scorer._kwargs if metric_params is None else metric_params
    else:
        # Metric given as a lambda function
        name = metric.__name__
        metric_fn = metric
        metric_params = {} if metric_params is None else metric_params

    return (name, metric_fn, metric_params)
