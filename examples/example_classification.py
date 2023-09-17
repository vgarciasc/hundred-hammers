# pylint: skip-file

from sklearn.datasets import load_iris

import logging
from hundred_hammers import HundredHammersClassifier, plot_batch_results, plot_confusion_matrix, hh_logger
from hundred_hammers.model_zoo import KNeighborsClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning


def main():
    data = load_iris()
    X = data.data
    y = data.target

    # Create the model
    hh = HundredHammersClassifier(show_progress_bar=True)

    # Evaluate the model
    df_results = hh.evaluate(X, y, optim_hyper=True, n_grid_points=4)

    # Print the results
    print(df_results)

    # Plot the results
    plot_batch_results(df_results, metric_name="ACC", title="Iris Dataset", display=True)
    plot_confusion_matrix(X, y, class_dict={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
                          model=KNeighborsClassifier(), title="Iris Dataset", display=True)

if __name__ == "__main__":
    # hh_logger.setLevel(logging.WARNING)
    # hh_logger.setLevel(logging.INFO)
    hh_logger.setLevel(logging.ERROR)

    main()
