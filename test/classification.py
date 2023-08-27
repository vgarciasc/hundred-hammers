from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from hundred_hammers.classifier import HundredHammersClassifier
from hundred_hammers.plots import plot_batch_results, plot_confusion_matrix

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target

    # Create the model
    # hh = HundredHammersClassifier(models=[("Dummy Classifier", DummyClassifier(), {}),
    #                                       ("Decision Tree", DecisionTreeClassifier(), {}),
    #                                       ("K Nearest Neighbors", KNeighborsClassifier(), {})])
    hh = HundredHammersClassifier()

    # Evaluate the model
    df_results = hh.evaluate(X, y)

    # Print the results
    print(df_results)

    # Plot the results
    plot_batch_results(df_results, metric_name="Accuracy", title="Iris Dataset")
    plot_confusion_matrix(X, y, class_dict={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
                          model=KNeighborsClassifier(), title="Iris Dataset")
