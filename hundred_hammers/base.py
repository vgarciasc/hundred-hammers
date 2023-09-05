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
from .metric_alias import metric_alias

class HundredHammersBase():
    def __init__(self, models=None, metrics=None, test_size=0.2, n_folds=5, n_folds_tune=5, n_seeds=10, verbose=True):
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

        self.test_size = test_size
        self.n_folds = n_folds
        self.n_folds_tune = n_folds_tune
        self.n_seeds = n_seeds
        self.verbose = verbose




    def calc_metrics(self, y_true, y_pred):
        """
        Calculate metrics for a given model.

        :param y_true: true values
        :param y_pred: predicted values
        :return: a list with the results for each metric
        """

        return [metric_fn(y_true, y_pred, **metric_params) for _, metric_fn, metric_params in self.metrics]

    def evaluate(self, X, y):
        """
        Evaluate all models on a given dataset.

        :param X: input observations
        :param y: target values
        :return: a DataFrame with the results
        """
        data = []
        for i, (name, model, cv_params) in enumerate(self.models):
            print(f"Running model [{i}/{len(self.models)}]: {name}") if self.verbose else None

            if cv_params:
                model = self.tune_model(X, y, model, cv_params)

            res = self.evaluate_model_multiple_seeds(X, y, model, n_evals=self.n_seeds)

            val = {"Model": name}
            for i, (metric_name, metric, _) in enumerate(self.metrics):
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    val[f"Avg {metric_name} ({data_name})"] = np.mean([m[i] for m in res[j]])
                    val[f"Std {metric_name} ({data_name})"] = np.std([m[i] for m in res[j]])

            data.append(val)

        df_results = pd.DataFrame(data)
        return df_results

    def evaluate_model_multiple_seeds(self, X, y, model, n_evals=10, should_print=False):
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
            res = self.evaluate_model(X, y, model, seed=seed)

            results_val_train += res[0]
            results_val_test += res[1]
            results_train.append(res[2])
            results_test.append(res[3])

        results = [results_val_train, results_val_test, results_train, results_test]

        if should_print:
            for i, metric in enumerate(self.metrics):
                print(f"{i}: {metric}")
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    avg_res = np.mean([m[i] for m in results[j]])
                    std_res = np.std([m[i] for m in results[j]])
                    print(f"\t{data_name}: {avg_res:.3f} ± {std_res:.3f}")
                print()

        return results

    def evaluate_model(self, X, y, model, seed=0):
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

        for train_index, test_index in kf.split(X_train, y_train):
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

    def tune_model(self, X, y, model, cv_params):
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

        clf = GridSearchCV(model, cv_params, scoring="neg_mean_squared_error", n_jobs=-1, cv=self.n_folds_tune)
        clf.fit(X, y)

        return clf.best_estimator_
