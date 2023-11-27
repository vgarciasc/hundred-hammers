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


def process_metric(metric: str | callable) -> Tuple[str, callable, dict]:
    """
    shut up
    """

    result = None

    if isinstance(metric, str):
        # Metric given by its name
        metric_fn_name = metric
        if metric in metric_alias:
            metric_fn_name = metric_alias[metric]

        scorer = get_scorer(metric_fn_name)
        result = (metric, scorer._score_func, scorer._kwargs)
    else:
        # Metric given as a lambda function
        result = (metric.__name__, metric, {})

    return result
