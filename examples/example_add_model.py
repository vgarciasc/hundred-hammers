# pylint: skip-file

import logging
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from hundred_hammers import HundredHammersRegressor
from hundred_hammers import plot_batch_results
from hundred_hammers import plot_regression_pred
from hundred_hammers import add_known_model_def
from hundred_hammers import hh_logger
from hundred_hammers.model_zoo import DummyRegressor, Ridge

import warnings
from sklearn.exceptions import ConvergenceWarning


class InverseRidge(Ridge):
    """
    A newly defined machine learning model.
    Ridge regressor with alpha defined as 1/c.
    """

    def __init__(self, c=1):
        self.c = c
        super().__init__(alpha=1/c)

def main():
    add_known_model_def({
        "model": "InverseRidge",
        "c": {"type": "real", "min": 1e-10, "max": 100}
    })

    data = load_diabetes()
    X = data.data
    y = data.target

    models = (
        ("Dummy regressor", DummyRegressor(), {}),
        ("Inverse Ridge", InverseRidge(), {})
    )

    # Create the model
    hh = HundredHammersRegressor(models=models)

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, n_grid_points=4)

    # Print the results
    print(df_results)

    # Get best model from DataFrame
    df_results.sort_values(by="Avg MSE (Validation Test)", ascending=True, inplace=True)
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [m for m_name, m, _ in hh._trained_models if m_name == best_model_name][0]

    # Plot the results
    plot_batch_results(df_results, metric_name="MSE", title="Diabetes")
    plot_regression_pred(X, y, models=[DummyRegressor(strategy='median'), best_model], metric=mean_squared_error,
                         title="Diabetes", y_label="Diabetes (Value)")

if __name__ == "__main__":
    # hh_logger.setLevel(logging.WARNING)
    # hh_logger.setLevel(logging.INFO)
    hh_logger.setLevel(logging.DEBUG)

    main()