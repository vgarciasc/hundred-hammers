from __future__ import annotations
from typing import Tuple, List, Iterable
import warnings
import random
from copy import deepcopy, copy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer, RobustScaler
from .config import hh_logger
from .metric_alias import metric_alias
from .hyperparameters import find_hyperparam_grid
from tqdm import tqdm


def _process_metric(metric: str | callable) -> Tuple[str, callable, dict]:
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


class HundredHammersBase:
    """
    Base HundredHammers class. 
    Implements methods for automatic machine learning like evaluating a list of models
    and performing hyperparameter optimization.

    :param models: List of models to evaluate.
    :param metrics: Metrics to use to evaluate the models.
    :param eval_metric: Target metric to use in hyperparameter optimization.
    :param test_size: Percentage of the dataset to use for testing.
    :param n_folds: Number of Cross Validation folds.
    :param n_folds_tune: Number of Cross Validation folds in grid search.
    :param n_evals: Number of times to repeat the training of the models.
    :param seed_strategy: Strategy used to generate the seeds for the different evaluations ('sequential' or 'random')
    """

    def __init__(
        self,
        models: Iterable[Tuple[str, BaseEstimator, dict]] = None,
        metrics: Iterable[str | callable] = None, 
        eval_metric: str | callable = None,
        input_transform: TransformerMixin | str = None,
        test_size: float = 0.2,
        n_folds: int = 5,
        n_folds_tune: int = 5,
        n_evals: int = 10,
        show_progress_bar: bool = True,
        seed_strategy: str = 'sequential'
    ):
        self.models = models
        self.metrics = [_process_metric(metric) for metric in metrics]

        if eval_metric is None:
            self.eval_metric = self.metrics[0]
        else:
            self.eval_metric = _process_metric(eval_metric)

        self.test_size = test_size
        self.n_folds = n_folds
        self.n_folds_tune = n_folds_tune
        self.n_evals = n_evals
        self.show_progress_bar = show_progress_bar
        self.seed_strategy = seed_strategy

        if input_transform:
            if isinstance(input_transform, TransformerMixin):
                input_transform = input_transform
            elif isinstance(input_transform, str):
                if input_transform == "MinMax":
                    input_transform = MinMaxScaler()
                elif input_transform == "MaxAbs":
                    input_transform = MaxAbsScaler()
                elif input_transform == "Standard":
                    input_transform = StandardScaler()
                elif input_transform == "Norm":
                    input_transform = Normalizer()
                elif input_transform == "Robust":
                    input_transform = RobustScaler()
        self._input_transform = input_transform

        self._report = pd.DataFrame()
        self._best_params = []
        self._trained_models = self.models

    @property
    def report(self) -> pd.DataFrame:
        """
        Pandas dataframe reflecting the results of the last evaluation of the models.

        :return: Dataframe with the performance of each of the models.
        """

        if self._report.empty:
            hh_logger.warning("No reports available. "
                              "Use the `evaluate` method to generate a report.")

        return self._report

    @property
    def best_params(self) -> List[Tuple[str, dict]]:
        """
        List of the best hyperparameters found for each model.

        :return: List of the best hyperparameters obtained for each model. 
        """

        if not self._best_params:
            hh_logger.warning("No available hyperparameters. "
                              "Hyperparameter optimization not performed.")

        model_names = [m_tup[0] for m_tup in self.models]

        return list(zip(model_names, self._best_params))

    @property
    def trained_models(self) -> Iterable[tuple[str, BaseEstimator, dict]]:
        """
        Get the trained models.

        :return: A list of models in the form of tuples (name, model, hyperparameters).
        """

        if self._report.empty:
            hh_logger.warning("The models were not trained, returning untrained models. "
                              "Use the `evaluate` method to train them.")

        return self._trained_models

    def _calc_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        """
        Calculate metrics for a given model.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :return: A list with the results for each metric.
        """

        return [metric_fn(y_true, y_pred, **metric_params) for _, metric_fn, metric_params in self.metrics]

    def evaluate(self, X: np.ndarray, y: np.ndarray, optim_hyper: bool = True,
                 n_grid_points: int = 10) -> pd.DataFrame:
        """
        Train every model to obtain its performance.

        :param X: Input data.
        :param y: Target data.
        :param optim_hyper: Whether to optimize the hyperparameters of the models.
        :param n_grid_points: Number of points to take for each hyperparameter in grid search.
        :return: Dataframe with the performance of each of the models.
        """

        # Do train/test split
        # TODO: set seed for the train/test split function
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=0)

        # Normalize inputs
        if self._input_transform:
            self._input_transform = self._input_transform.fit(X_train)
            X_norm_train = self._input_transform.transform(X_train)
            X_norm_test = self._input_transform.transform(X_test)
        else:       
            X_norm_train = X_train
            X_norm_test = X_test

        # Do (or don't do) hyperparameter optimization
        if optim_hyper:
            new_models = self.tune_models(X_norm_train, y_train, n_grid_points)
        else:
            new_models = deepcopy(self.models)

        # Evaluate models
        report, trained_models = self._evaluate_models(X_norm_train, y_train, X_norm_test, y_test, new_models)

        # Add normalization to models with pipelines
        if self._input_transform:
            trained_models = [Pipeline([('scaler', self._input_transform), ('model', model),]) for model in trained_model]

        # Store data in the object's attributes
        self._report = report
        self._trained_models = [(m_name, tmodel, param_grid) for (m_name, _, param_grid), tmodel in zip(self.models, trained_models)]

        return report

    def optimize_hyperparams(self, X: np.ndarray, y: np.ndarray, n_grid_points: int = 10) -> List[dict]:
        """
        Obtain the best set of parameters for each of the models.

        :param X: Input data.
        :param y: Target data.
        :param n_grid_points: Number of points to take for each hyperparameter in grid search.
        :return: List of the best hyperparameters obtained for each model. 
        """

        self._best_params = []

        for name, model, param_grid in tqdm(self.models, desc="Optimizing hyperparameters...",
                                            leave=False, disable=not self.show_progress_bar):
            best_params_model = self._optimize_model_hyperparams(X, y, model, param_grid, n_grid_points)
            self._best_params.append(best_params_model)

        return self._best_params

    def tune_models(self, X: np.ndarray, y: np.ndarray,
                    n_grid_points: int = 10) -> List[Tuple[str, BaseEstimator, dict]]:
        """
        Tune a model using cross-validation.

        :param X: Input observations.
        :param y: Target values.
        :param n_grid_points: Number of points to take for each hyperparameter in grid search.
        :return: The tuned model.
        """

        best_param_list = self.optimize_hyperparams(X, y, n_grid_points)

        new_models = []
        for (model_name, model, model_param_grid), best_params in zip(self.models, best_param_list):
            # change the parameters without overwriting the model
            configured_model = copy(model).set_params(**best_params)

            new_models.append((model_name, configured_model, model_param_grid))

        return new_models

    def _evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                         models: Iterable[Tuple[str, BaseEstimator, dict]]) -> Tuple[pd.DataFrame, list[BaseEstimator]]:
        """
        Evaluate all models on a given dataset with their default hyperparameters.
        
        :param X: Input observations.
        :param y: Target values.
        :return: A DataFrame with the results.
        """

        data = []
        trained_models = []

        for i, (name, model, _) in enumerate(tqdm(models, desc="Evaluating models...",
                                                  disable=not self.show_progress_bar)):
            hh_logger.info(f"Running model [{i+1}/{len(models)}]: {name}")

            res, new_model = self._evaluate_model_cv_multiple_seeds(X_train, y_train, X_test, y_test, model, n_evals=self.n_evals)
            trained_models.append(new_model)

            val = {"Model": name}
            for i, (metric_name, _, _) in enumerate(self.metrics):
                for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                    val[f"Avg {metric_name} ({data_name})"] = np.mean([m[i] for m in res[j]])
                    val[f"Std {metric_name} ({data_name})"] = np.std([m[i] for m in res[j]])

            data.append(val)

        return pd.DataFrame(data), trained_models

    def _evaluate_model_cv_multiple_seeds(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                                          model: BaseEstimator, n_evals: int = 10) -> Tuple[list[list[float]], BaseEstimator]:
        """
        Evaluate a model multiple times, with a different seed every time.

        :param X: Input observations.
        :param y: Target values.
        :param model: Model to evaluate.
        :param n_evals: Number of times to train the model (each iteration uses a different seed).
        :return: A tuple with the results for validation train, validation test, train and test.
        """

        results_val_train, results_val_test = [], []
        results_train, results_test = [], []

        if self.seed_strategy == "random":
            seeds = np.random.randint(0, 1000000, n_evals)
        elif self.seed_strategy == "sequential":
            seeds = range(0, n_evals)
        else:
            raise ValueError(f"Unknown seed strategy: {self.seed_strategy}")

        # take `n_evals` different seeds
        for i, seed in enumerate(tqdm(seeds, desc=f"        {model.__class__.__name__}",
                                      leave=False, disable=not self.show_progress_bar)):
            hh_logger.debug(f"Iteration [{i+1}/{n_evals-1}]")
            result_val_train, result_val_test, result_train, result_test, trained_model = self._evaluate_model_cv(
                X_train, y_train, X_test, y_test, model, seed=seed
            )

            results_val_train += result_val_train
            results_val_test += result_val_test
            results_train.append(result_train)
            results_test.append(result_test)

        results = [results_val_train, results_val_test, results_train, results_test]

        model_info = "Metrics:\n"
        for i, (metric_name, _, _) in enumerate(self.metrics):
            model_info += f"{i}: {metric_name}\n"
            for j, data_name in enumerate(["Validation Train", "Validation Test", "Train", "Test"]):
                avg_res = np.mean([m[i] for m in results[j]])
                std_res = np.std([m[i] for m in results[j]])
                model_info += f"\t{data_name}: {avg_res:.3f} ± {std_res:.3f}\n"
            model_info += "\n"
        hh_logger.info(model_info)

        return results, trained_model

    def _evaluate_model_cv(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model: BaseEstimator,
                           seed: int = 0) -> tuple[list[list[float]], list[list[float]], list[float], list[float], BaseEstimator]:
        """
        Evaluate a model on a given dataset.

        :param X: Input observations.
        :param y: Target values.
        :param model: Model to evaluate.
        :param seed: Random seed.
        :return: A tuple with the results for validation train, validation test, train and test.
        """

        if hasattr(model, 'random_state'):
            model.random_state = seed

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

        results_val_train, results_val_test = [], []

        for split_idx, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            val_model = copy(model)
            hh_logger.debug(f"Split [{split_idx}/{self.n_folds}]")

            X_val_train, X_val_test = X_train[train_index], X_train[test_index]
            y_val_train, y_val_test = y_train[train_index], y_train[test_index]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_model.fit(X_val_train, y_val_train)

            result_val_train = self._calc_metrics(y_val_train, val_model.predict(X_val_train))
            result_val_test = self._calc_metrics(y_val_test, val_model.predict(X_val_test))

            results_val_train.append(result_val_train)
            results_val_test.append(result_val_test)

        trained_model = copy(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained_model.fit(X_train, y_train)

        result_train = self._calc_metrics(y_train, trained_model.predict(X_train))
        result_test = self._calc_metrics(y_test, trained_model.predict(X_test))

        return results_val_train, results_val_test, result_train, result_test, trained_model

    def _optimize_model_hyperparams(self, X: np.ndarray, y: np.ndarray,
                                    model: BaseEstimator, param_grid: dict = None,
                                    n_grid_points: int = 10) -> dict:
        """
        Optimize the hyperparameters of a model.
        
        :param X: Input data.
        :param y: Target data.
        :param model: Model to optimize.
        :param param_grid: Predefined hyperparameter grid.
        :param n_grid_points: Number of points to take for each hyperparameter in grid search.
        :return: The best hyperparameters found for the model.
        """

        if not param_grid:
            hh_logger.info(f"No specified hyperparameter grid for {type(model).__name__}."
                           " Generating hyperparameter grid.")
            param_grid = find_hyperparam_grid(model, n_grid_points)

        eval_metric = lambda y_true, y_pred: self.eval_metric[1](y_true, y_pred,
                                                                 **self.eval_metric[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search_model = GridSearchCV(model, param_grid, scoring=eval_metric,
                                             n_jobs=-1, cv=self.n_folds_tune)
            grid_search_model.fit(X, y)

        results = pd.DataFrame(grid_search_model.cv_results_)
        results.dropna()
        best_params_df = results[results["rank_test_score"] == results["rank_test_score"].min()]
        best_params = best_params_df.head(1)['params'][0]

        return best_params