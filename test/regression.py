from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from hundred_hammers.regressor import HundredHammersRegressor
from hundred_hammers.plots import plot_regression_pred

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    data = load_diabetes()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersRegressor(models=[("Linear Regression", LinearRegression(), {}),
                                            ("Dummy Regressor", DummyRegressor(), {})],)

    # Evaluate the model
    df_results = hh.evaluate(X, y)

    # Print the results
    print(df_results)

    # Plot the results
    plot_regression_pred(X, y, models=[LinearRegression(), DecisionTreeRegressor(ccp_alpha=0.5)],
                         metric=mean_squared_error, title="Diabetes", y_label="Diabetes (Value)")
