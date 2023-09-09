import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import get_scorer
from adjustText import adjust_text
from sklearn.model_selection import GridSearchCV
from .config import hh_logger
from .metric_alias import metric_alias
from .hyperparameters import find_hyperparam_grid
import warnings


def _process_metric(metric):
    result = None

    if isinstance(metric, str):
        # Metric given by its name
        metric_fn_name = metric
        if metric in metric_alias:
            metric_fn_name = metric_alias[metric]

        scorer = get_scorer(metric_fn_name)
        result = (metric, scorer._score_func, scorer._kwargs)
    else:
        # Metric given as a lambda function
        result = (metric.__name__, metric, {})

    return result


class HundredHammersBase():
    def __init__(self, models=None, metrics=None, eval_metric=None, test_size=0.2,
                 n_folds=5, n_folds_tune=5, n_seeds=10, verbose=True):
        self.models = models
        self.metrics = [_process_metric(metric) for metric in metrics]
        
        if eval_metric is None:
            self.eval_metric = self.metrics[0]
        else:
            self.eval_metric = _process_metric(eval_metric)

        self.test_size = test_size
        self.n_folds = n_folds
        self.n_folds_tune = n_folds_tune
        self.n_seeds = n_seeds
        self.verbose = verbose
        self._report = pd.DataFrame()
        self._best_params = []
    
    @property
    def report(self):
        if self._report.empty:
            hh_logger.warn("No reports available. Use the `evaluate` method to generate a report.")
        
        return self._report
    
    @property
    def best_params(self):
        if not self._best_params:
            hh_logger.warn("No available hyperparameters. Hyperparameter optimization not performed.")
        
        model_names = [m_tup[0] for m_tup in self.models]

        return list(zip(model_names, self._best_params))

    def _calc_metrics(self, y_true, y_pred):
        """
        Calculate metrics for a given model.

        :param y_true: true values
        :param y_pred: predicted values
        :return: a list with the results for each metric
        """

        return [metric_fn(y_true, y_pred, **metric_params) for _, metric_fn, metric_params in self.metrics]
    
    def evaluate(self, X, y, optim_hyper=True, n_grid_points=10):
        """
        """

        if optim_hyper:
            self.models = self.tune_models(X, y, n_grid_points)
        
        report = self._evaluate_models(X, y)
        
        self._report = report

        return report
    
    def optimize_hyperparams(self, X, y, n_grid_points=10):
        """
        Obtain the best set of parameters for each of the models.

        :param X:
        :param y:
        :param n_grid_points:

        :return:
        """

        self._best_params = [self._optimize_model_hyperparams(X, y, model, param_grid, n_grid_points) for _, model, param_grid in self.models]

        return self._best_params
    
    def tune_models(self, X, y, n_grid_points=10):
        """
        Tune a model using cross-validation.

        :param X: input observations
        :param y: target values
        :param model: model to tune
        :param cv_params: parameters to tune (as in GridSearchCV)
        :return: the tuned model
        """

        best_param_list = self.optimize_hyperparams(X, y, n_grid_points)

        new_models = []
        for (model_name, model, model_param_grid), best_params in zip(self.models, best_param_list):
            new_models.append((model_name, model.set_params(**best_params), model_param_grid))
        
        return new_models

    def _evaluate_models(self, X, y):
        """
        Evaluate all models on a given dataset with their default hyperparameters.
        
        :param X: input observations
        :param y: target values
        :return: a DataFrame with the results
        """
        
        data = []
        for i, (name, model, _) in enumerate(self.models):
            hh_logger.info(f"Running model [{i}/{len(self.models)}]: {name}")

            res = self._evaluate_model_cv_multiple_seeds(X, y, model, n_evals=self.n_seeds)

            val = {"Model": name}
            for i, (metric_name, metric, _) in enumerate(self.metrics):
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    val[f"Avg {metric_name} ({data_name})"] = np.mean([m[i] for m in res[j]])
                    val[f"Std {metric_name} ({data_name})"] = np.std([m[i] for m in res[j]])

            data.append(val)

        return pd.DataFrame(data)

        # if cv_params:
        #         model = self.tune_model(X, y, model, cv_params)

    def _evaluate_model_cv_multiple_seeds(self, X, y, model, n_evals=10):
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
            res = self._evaluate_model_cv(X, y, model, seed=seed)

            results_val_train += res[0]
            results_val_test += res[1]
            results_train.append(res[2])
            results_test.append(res[3])

        results = [results_val_train, results_val_test, results_train, results_test]

        if self.verbose:
            for i, (metric_name, _, _) in enumerate(self.metrics):
                print(f"{i}: {metric_name}")
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    avg_res = np.mean([m[i] for m in results[j]])
                    std_res = np.std([m[i] for m in results[j]])
                    print(f"\t{data_name}: {avg_res:.3f} ± {std_res:.3f}")
                print()

        return results

    def _evaluate_model_cv(self, X, y, model, seed=0):
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

            result_val_train = self._calc_metrics(model.predict(X_val_train), y_val_train)
            result_val_test = self._calc_metrics(model.predict(X_val_test), y_val_test)

            results_val_train.append(result_val_train)
            results_val_test.append(result_val_test)

        model.fit(X_train, y_train)
        result_train = self._calc_metrics(model.predict(X_train), y_train)
        result_test = self._calc_metrics(model.predict(X_test), y_test)

        return results_val_train, results_val_test, result_train, result_test
    
    def _optimize_model_hyperparams(self, X, y, model, param_grid=None, n_grid_points=10):
        """
        """

        if not param_grid:
            hh_logger.info(f"No predefined hyperparameter grid for {type(model).__name__}. Generating hyperparameter grid.")
            param_grid = find_hyperparam_grid(model, n_grid_points)
        
        eval_metric = lambda y_true, y_pred: self.eval_metric[1](y_true, y_pred, **self.eval_metric[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search_model = GridSearchCV(model, param_grid, scoring=eval_metric, n_jobs=-1, cv=self.n_folds_tune)
            grid_search_model.fit(X, y)

        cv_results = pd.DataFrame(grid_search_model.cv_results_)
        cv_results.dropna()
        best_params = cv_results[cv_results["rank_test_score"] == cv_results["rank_test_score"].min()].head(1)['params'][0]

        return best_params