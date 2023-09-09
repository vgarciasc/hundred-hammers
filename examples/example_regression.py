import logging
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from hundred_hammers import HundredHammersRegressor, plot_batch_results, plot_regression_pred
from hundred_hammers.model_zoo import DummyRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning


def main():
    data = load_diabetes()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersRegressor()

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=False)

    # Print the results
    print(df_results)

    # Get best model from DataFrame
    df_results.sort_values(by="Avg MSE (Validation Test)", ascending=True, inplace=True)
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [m for m_name, m, _ in hh.models if m_name == best_model_name][0]

    # Plot the results
    plot_batch_results(df_results, metric_name="MSE", title="Diabetes")
    plot_regression_pred(X, y, models=[DummyRegressor(strategy='median'), best_model], metric=mean_squared_error,
                         title="Diabetes", y_label="Diabetes (Value)")

if __name__ == "__main__":
    main()