# Hundred Hammers

> "At least *one* of them is bound to do the trick."

*Hundred Hammers* is a Python package that helps you batch-test ML models in a dataset. It can be used out-of-the-box
to run most popular ML models and metrics, or it can be easily extended to include your own.

- Supports both classification and regression.
- Already comes strapped with most sci-kit learn models.
- Already comes with several plots to visualize the results.
- Easy to integrate with parameter tuning from GridSearch CV.
- Already gives you the average metrics from training, test, validation (train) and validation (test) sets.
- Allows you to define how many seeds to consider, so you can increase the significance of your results.
- Produces a Pandas DataFrame with the results (which can be exported to CSV and analyzed elsewhere).

## Installation

Clone the repository and run `pip install .` in the root directory. 

## Examples

Full examples can be found in the `test` directory. Here's a simple example of how to use Hundred Hammers to run a
batch classification on Iris data:

```python
from hundred_hammers.classifier import HundredHammersClassifier
from hundred_hammers.plots import plot_batch_results
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

hh = HundredHammersClassifier()
df_results = hh.evaluate(X, y)

plot_batch_results(df_results, metric_name="Accuracy", title="Iris Dataset")
```

This already gives us a DataFrame with the results from several different models, and a nice plot of the results:

<img src="multimedia/iris_batch.png" width=600px;>

### Other plots

We can also use Hundred Hammers to produce nice confusion matrices plots and regression predictions:

```python
from hundred_hammers.plots import plot_confusion_matrix
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris()
X, y = data.data, data.target
plot_confusion_matrix(X, y, class_dict={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
                      model=DecisionTreeClassifier(), title="Iris Dataset")
```

<img src="multimedia/iris_cm.png" width=600px;>

```python
from hundred_hammers.plots import plot_regression_pred
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

data = load_diabetes()
X, y = data.data, data.target
plot_regression_pred(X, y, models=[DummyRegressor(strategy='median'), best_model], metric=mean_squared_error,
                     title="Diabetes", y_label="Diabetes (Value)")
```

<img src="multimedia/diabetes_pred.png" width=600px;>

Finally, it is also possible to compare different datasets and compare their results (each dot is a model).

```python
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
```

<img src="multimedia/dataset_batch.png" width=600px;>