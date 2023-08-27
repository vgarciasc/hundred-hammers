from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from hundred_hammers.regressor import HundredHammersRegressor
from hundred_hammers.plots import plot_batch_results, plot_regression_pred

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    data = load_diabetes()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersRegressor()

    # Evaluate the model
    df_results = hh.evaluate(X, y)

    # Print the results
    print(df_results)

    # Get best model from DataFrame
    df_results.sort_values(by="Avg MSE (Validation Test)", ascending=True, inplace=True)
    best_model_name = df_results.iloc[0]["Model"]
    best_model = [m for m_name, m, _ in hh.models if m_name == best_model_name][0]

    # Plot the results
    plot_batch_results(df_results, metric_name="MSE", title="Diabetes")
    plot_regression_pred(X, y, models=[best_model], metric=mean_squared_error,
                         title="Diabetes", y_label="Diabetes (Value)")
