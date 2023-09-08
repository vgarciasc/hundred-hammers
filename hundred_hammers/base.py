import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import pearsonr
import math
from sklearn.model_selection import train_test_split, KFold
from adjustText import adjust_text
from sklearn.model_selection import GridSearchCV
from .config import hh_logger
from .metric_alias import metric_alias

class HundredHammersBase():
    def __init__(self, models=None, metrics=None, eval_metric=None, test_size=0.2, n_folds=5, n_folds_tune=5, n_seeds=10, verbose=True):
        self.models = models
        
        self.metrics = []

        for metric in metrics:
            if isinstance(metric, str):
                # Metric given by its name
                metric_fn_name = metric
                if metric in metric_alias:
                    metric_fn_name = metric_alias[metric]

                scorer = get_scorer(metric_fn_name)
                self.metrics.append((scorer._score_func, scorer._kwargs))
            else:
                # Metric given as a lambda function
                self.metrics.append((metric, {}))
        
        if eval_metric is None:
            self.eval_metric = self.metrics[0]

        self.test_size = test_size
        self.n_folds = n_folds
        self.n_folds_tune = n_folds_tune
        self.n_seeds = n_seeds
        self.verbose = verbose
        self.report = None

    def calc_metrics(self, y_true, y_pred):
        """
        Calculate metrics for a given model.

        :param y_true: true values
        :param y_pred: predicted values
        :return: a list with the results for each metric
        """

        return [metric_fn(y_true, y_pred, **metric_params) for _, metric_fn, metric_params in self.metrics]
    
    def evaluate(self, X, y, optim_hyper=True):
        report = None
        if optim_hyper:
            report = evaluate_tune(self, X, y)
        else:
            report = evaluate_default(self, X, y)
        
        self.report = report

        return report

    def evaluate_default(self, X, y):
        """
        Evaluate all models on a given dataset with their default hyperparameters.
        
        :param X: input observations
        :param y: target values
        :return: a DataFrame with the results
        """
        
        data = []
        for i, (name, model, _) in enumerate(self.models):
            hh_logger.info(f"Running model [{i}/{len(self.models)}]: {name}")

            res = self.evaluate_model_cv_multiple_seeds(X, y, model, n_evals=self.n_seeds)

            val = {"Model": name}
            for i, (metric_name, metric, _) in enumerate(self.metrics):
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    val[f"Avg {metric_name} ({data_name})"] = np.mean([m[i] for m in res[j]])
                    val[f"Std {metric_name} ({data_name})"] = np.std([m[i] for m in res[j]])

            data.append(val)

        return pd.DataFrame(data)


    def evaluate_tune(self, X, y):
        """
        Evaluate all models on a given dataset and optimize their hyperparameters.

        :param X: input observations
        :param y: target values
        :return: a DataFrame with the results
        """

        data = []
        for i, (name, model, cv_params) in enumerate(self.models):
            hh_logger.info(f"Running model [{i}/{len(self.models)}]: {name}")

            if cv_params:
                model = self.tune_model(X, y, model, cv_params)

            res = self.evaluate_model_cv_multiple_seeds(X, y, model, n_evals=self.n_seeds)

            val = {"Model": name}
            for i, (metric_name, metric, _) in enumerate(self.metrics):
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    val[f"Avg {metric_name} ({data_name})"] = np.mean([m[i] for m in res[j]])
                    val[f"Std {metric_name} ({data_name})"] = np.std([m[i] for m in res[j]])

            data.append(val)

        return pd.DataFrame(data)

    def evaluate_model_cv_multiple_seeds(self, X, y, model, n_evals=10):
        """
        Evaluate a model multiple times with different seeds.

        :param X: input observations
        :param y: target values
        :param model: model to evaluate
        :param n_evals: how many different seeds to use
        :param should_print: whether to print the results
        :return: a tuple with the results for validation train, validation test, train and test
        """

        results_val_train, results_val_test = [], []
        results_train, results_test = [], []

        for seed in range(0, n_evals):
            hh_logger.info(f"Iteration [{seed}/{n_evals}]")
            res = self.evaluate_model_cv(X, y, model, seed=seed)

            results_val_train += res[0]
            results_val_test += res[1]
            results_train.append(res[2])
            results_test.append(res[3])

        results = [results_val_train, results_val_test, results_train, results_test]

        if self.verbose:
            for i, metric in enumerate(self.metrics):
                print(f"{i}: {metric}")
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    avg_res = np.mean([m[i] for m in results[j]])
                    std_res = np.std([m[i] for m in results[j]])
                    print(f"\t{data_name}: {avg_res:.3f} ± {std_res:.3f}")
                print()

        return results

    def evaluate_model_cv(self, X, y, model, seed=0):
        """
        Evaluate a model on a given dataset.

        :param X: input observations
        :param y: target values
        :param model: model to evaluate
        :param seed: random seed
        :return: a tuple with the results for validation train, validation test, train and test
        """

        if hasattr(model, 'random_state'):
            model.random_state = seed

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=seed)

        results_val_train, results_val_test = [], []

        for split_idx, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            hh_logger.debug(f"Split [{split_idx}/{self.n_folds}]")

            X_val_train, X_val_test = X_train[train_index], X_train[test_index]
            y_val_train, y_val_test = y_train[train_index], y_train[test_index]

            model.fit(X_val_train, y_val_train)

            result_val_train = self.calc_metrics(model.predict(X_val_train), y_val_train)
            result_val_test = self.calc_metrics(model.predict(X_val_test), y_val_test)

            results_val_train.append(result_val_train)
            results_val_test.append(result_val_test)

        model.fit(X_train, y_train)
        result_train = self.calc_metrics(model.predict(X_train), y_train)
        result_test = self.calc_metrics(model.predict(X_test), y_test)

        return results_val_train, results_val_test, result_train, result_test

    def tune_model(self, X, y, model, cv_params=None):
        """
        Tune a model using cross-validation.

        :param X: input observations
        :param y: target values
        :param model: model to tune
        :param cv_params: parameters to tune (as in GridSearchCV)
        :return: the tuned model
        """

        if cv_params is None:
            return model
        
        hh_logger.debug(f"Optimizing hyperparameters.")

        clf = GridSearchCV(model, cv_params, scoring=self.eval_metric, n_jobs=-1, cv=self.n_folds_tune)
        clf.fit(X, y)

        return clf.best_estimator_


def construct_hyperparam_grid(hyperparam_grid_template: List[dict], n_grid_points: int = 10) -> List[dict]:
    """
    Generate a grid of hyperparameters from their definition.

    Parameters
    ----------
    hyperparam_grid_template: List[dict]
        Definition of the hyperparameters to be generated as a grid.
    n_grid_points: int, optional
        Number of values to pick for each hyperparameter.
    
    Returns
    -------
    hyperparameter_grid: List[dict]
        List of hyperparameter grids to use in grid search.
    """

    param_list = []
    for idx, hp_template in enumerate(hyperparam_grid_template):        
        keys = list(hp_template.keys())
        keys.remove("model")

        model_params = {}
        for k in keys:
            if hp_template[k]["type"] == "integer":
                model_params[k] = np.unique(np.round(np.linspace(hp_template[k]["min"], hp_template[k]["max"], n_grid_points))).astype(int)
            
            elif hp_template[k]["type"] == "real":
                model_params[k] = np.geomspace(max(hp_template[k]["min"], 1e-10), hp_template[k]["max"], n_grid_points)
            
            elif hp_template[k]["type"] == "categorical":
                model_params[k] = hp_template[k]["values"]

        param_list.append(model_params)

    return param_list
