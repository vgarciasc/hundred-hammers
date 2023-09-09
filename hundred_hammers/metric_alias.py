"""
This module provides some alternative names for metrics
implemented in sklearn.
"""

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