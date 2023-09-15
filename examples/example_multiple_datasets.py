# pylint: skip-file
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from hundred_hammers import HundredHammersClassifier
from hundred_hammers.plots import plot_multiple_datasets


def main():
    data = load_iris()
    X, y = data.data, data.target

    hh = HundredHammersClassifier()

    df = []
    for i, feature_name in enumerate(data.feature_names):
        X_i = X[:, [j for j in range(X.shape[1]) if j != i]]

        for degree in range(8):
            df_i = hh.evaluate(X_i ** degree, y, optim_hyper=False)
            df_i["Dataset"] = f"$X^{degree}$, w/out $x_{i}$"
            df.append(df_i)

    df_results = pd.concat(df, ignore_index=True)
    plot_multiple_datasets(df_results, metric_name="Avg ACC (Validation Test)", id_col="Dataset", title="Iris Dataset", display=True)

if __name__ == "__main__":
    main()
