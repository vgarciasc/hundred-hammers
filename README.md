<p align="center">
  <img src="multimedia/logo.png">
</p>

<p align="center">
    <a href="https://github.com/vgarciasc/hundred-hammers/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/vgarciasc/hundred-hammers?color=purple&style=plastic" /></a>
    <a href="https://github.com/vgarciasc/hundred-hammers/tree/master/test">
        <img src="https://img.shields.io/badge/coverage-100%25-green?color=green&style=plastic" /></a>
    <a href="https://github.com/vgarciasc/hundred-hammers/blob/master/build/requirements.txt">
        <img src="https://img.shields.io/badge/requirements-python3.8-red?color=blue&style=plastic" /></a>
    <a href="https://hundred-hammers.readthedocs.io/en/latest/">
        <img src="https://img.shields.io/badge/doc-readthedocs-green?color=orange&style=plastic" /></a>
    <a href="https://github.com/vgarciasc/hundred-hammers/releases/tag/TensorCRO-1.2.1">
        <img src="https://img.shields.io/pypi/v/hundred_hammers"></a>
</p>

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

The recommended way to install the library is through `pip install hundred_hammers`. However, if you want to fiddle around with the repo yourself, you can clone this repository, and run `pip install -e hundred_hammers/`

## Documentation

The documentation can be found in [ReadTheDocs](https://hundred-hammers.readthedocs.io/en/latest/). Code is formatted using Black with line length 150.

## Examples

Full examples can be found in the `examples` directory. As an appetizer, here's a simple one of how to use Hundred Hammers to run a
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

![](multimedia/iris_batch.png)

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

![](multimedia/iris_cm.png)

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

![](multimedia/diabetes_pred.png)

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

![](multimedia/dataset_batch.png)

## How is the data used?

By default, Hundred Hammers will split the data into train and test.
If the user defines a normalization procedure (through the `input_transform` parameter), then 
normalization will be fitted to the training data and applied to both partitions.
Next, if the user enabled hyperparameter optimization, the training data is used 
to fit the hyperparameters of each model, through a Grid Search with `n_folds_tune` folds.
The model is then trained on the training data and evaluated on both partitions 
to produce the **train** and **test** results.

As is standard in ML, the training data is also used in a cross-validation fashion,
according to the cross-validator passed by the user (through the `cross_validator` parameter).
The user-defined metrics are then average over the cross-validation folds to produced 
both **validation train** and **validation test** results.

Two DataFrames are provided to the user: a *full report* (`hh._full_report`) with the results for each 
model, seed, and cross-validation fold; and a *summary report* (`hh._report)`with the average results
for each model. 

Furthermore, with flexibility in mind, Hundred Hammers also allows the user to define 
how many seeds will be tested and averaged for both training and validation splitting. 
This is done through the `n_train_evals` and `n_val_evals` parameters, which are both `1`
by default (i.e. a single train/test split is done, and inside the training data, a 
single cross-validation scheme is run).

Since the usage of data is key, we provide below an image to illustrate how the data is used:

![](multimedia/data_flow.png)
