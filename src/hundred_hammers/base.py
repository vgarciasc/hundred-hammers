from __future__ import annotations
from typing import Tuple, List, Iterable, Any
import warnings
from copy import deepcopy, copy
import pandas as pd
import numpy as np
import rich
import time
from rich.progress import Progress
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.pipeline import Pipeline
from .config import hh_logger
from .metric_alias import process_metric
from .hyperoptimizer import HyperOptimizer, HyperOptimizerGridSearch


class HundredHammersBase:
    """
    Base HundredHammers class.
    Implements methods for automatic machine learning like evaluating a list of models
    and performing hyperparameter optimization.

    :param models: List of models to evaluate.
    :type models: Iterable[Tuple[str, BaseEstimator, dict]]
    :param metrics: Metrics to use to evaluate the models.
    :type metrics: Iterable[str | callable]
    :param eval_metric: Target metric to use in hyperparameter optimization.
    :type eval_metric: str | callable
    :param input_transform: Input normalization strategy used. Specified as a string or the normalization class. ('MinMax', 'MaxAbs', 'Standard', 'Norm', 'Robust')
    :type input_transform: TransformerMixin
    :param cross_validator: Cross Validator to use in the evaluation.
    :type cross_validator: callable
    :param cross_validator_params: Parameters for the Cross Validator.
    :type cross_validator_params: dict
    :param test_size: Percentage of the dataset to use for testing.
    :type test_size: float
    :param n_train_evals: Number of times to vary the training/test separation seed.
    :type n_train_evals: int
    :param n_val_evals: Number of times to vary the cross-validation seed.
    :type n_val_evals: int
    :param seed_strategy: Strategy used to generate the seeds for the different evaluations ('sequential' or 'random')
    :type seed_strategy: str
    """

    def __init__(
        self,
        models: Iterable[Tuple[str, BaseEstimator, dict]] = None,
        metrics: Iterable[str | callable] = None,
        eval_metric: str | callable = None,
        input_transform: TransformerMixin | str = None,
        cross_validator: callable = None,
        cross_validator_params: dict = None,
        test_size: float = 0.2,
        n_train_evals: int = 1,
        n_val_evals: int = 1,
        show_progress_bar: bool = True,
        seed_strategy: str = "sequential",
    ):
        self.models = models
        self.metrics = [process_metric(metric) for metric in metrics]

        if eval_metric is None:
            self.eval_metric = self.metrics[0]
        else:
            self.eval_metric = process_metric(eval_metric)

        self.cross_validator = cross_validator
        self.cross_validator_params = cross_validator_params
        self.test_size = test_size
        self.n_train_evals = n_train_evals
        self.n_val_evals = n_val_evals
        self.show_progress_bar = show_progress_bar
        self.seed_strategy = seed_strategy

        if input_transform:
            if isinstance(input_transform, type):
                input_transform = input_transform.__call__()
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
                else:
                    raise ValueError(
                        "Normalization method not implemented,"
                        "choose one of ['MinMax','MaxAbs','Standard','Norm','Robust'] or use one of sklearn's normalization classes."
                    )
            elif not isinstance(input_transform, TransformerMixin):
                raise ValueError("The input_transform must be either `None` or of type TransformerMixin or str.")

        self.model_name_str_pad = 25

        self._input_transform = input_transform

        self._full_report = pd.DataFrame()
        self._report = pd.DataFrame()
        self._best_params = []
        self._trained_models = self.models

    @property
    def full_report(self) -> pd.DataFrame:
        """
        Pandas dataframe reflecting the results of the last evaluation of the models with extra information.

        :return: Dataframe with the performance of each of the models.
        :rtype: DataFrame
        """

        if self._full_report.empty:
            hh_logger.warning("No reports available. Use the `evaluate` method to generate the full report.")

        return self._full_report

    @property
    def report(self) -> pd.DataFrame:
        """
        Pandas dataframe reflecting the results of the last evaluation of the models.

        :return: Dataframe with the performance of each of the models.
        :rtype: DataFrame
        """

        if self._report.empty:
            hh_logger.warning("No reports available. Use the `evaluate` method to generate a report.")

        return self._report

    @property
    def best_params(self) -> List[Tuple[str, dict]]:
        """
        List of the best hyperparameters found for each model.

        :return: List of the best hyperparameters obtained for each model.
        :rtype: List[Tuple[str, dict]]
        """

        if not self._best_params:
            hh_logger.warning("No available hyperparameters. Hyperparameter optimization not performed.")

        model_names = [m_tup[0] for m_tup in self.models]

        return list(zip(model_names, self._best_params))

    @property
    def trained_models(self) -> Iterable[tuple[str, BaseEstimator, dict]]:
        """
        Get the trained models.

        :return: A list of models in the form of tuples (name, model, hyperparameters).
        :rtype: Iterable[tuple[str, BaseEstimator, dict]]
        """

        if self._report.empty:
            hh_logger.warning("The models were not trained, returning untrained models. Use the `evaluate` method to train them.")

        return self._trained_models

    def _calc_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        """
        Calculate metrics for a given model.

        :param y_true: True values.
        :type y_true: ndarray
        :param y_pred: Predicted values.
        :type y_pred: ndarray
        :return: A list with the results for each metric.
        :rtype: List[float]
        """

        return [metric_fn(y_true, y_pred, **metric_params) for _, metric_fn, metric_params in self.metrics]

    def evaluate(self, X: np.ndarray, y: np.ndarray, optim_hyper: bool = True, hyperoptimizer: HyperOptimizer = None) -> pd.DataFrame:
        result = None
        if self.show_progress_bar:
            with Progress() as progress:
                result = self._evaluate(X, y, optim_hyper, hyperoptimizer, progress)
                progress.refresh()
        else:
            result = self._evaluate(X, y, optim_hyper, hyperoptimizer, progress)

        return result

    def _evaluate(
        self, X: np.ndarray, y: np.ndarray, optim_hyper: bool = True, hyperoptimizer: HyperOptimizer = None, progress: rich.progress.Progress = None
    ) -> pd.DataFrame:
        """
        Train every model to obtain its performance.

        :param X: Input data.
        :type X: ndarray
        :param y: Target data.
        :type y: ndarray
        :param optim_hyper: Whether to optimize the hyperparameters of the models.
        :type optim_hyper: bool
        :param hyperoptimizer: Hyperparameter optimizer that will find the best parameters for each model.
            By default, will use Grid Search with 5-fold cross validation on the evaluation metric.
        :type hyperoptimizer: HyperOptimizer
        :return: Dataframe with the performance of each of the models.
        :rtype: DataFrame
        """

        if optim_hyper and hyperoptimizer is None:
            hyperoptimizer = HyperOptimizerGridSearch(self.eval_metric)

        report = []
        seeds = self._generate_seeds(self.n_train_evals, self.seed_strategy)

        if progress is not None:
            seed_progress = progress.add_task("[cyan]Evaluating different train/test splits", total=len(seeds))

        for i, seed in enumerate(seeds):
            # Do train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=seed, stratify=self._stratify_array(y))

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
                new_models = self.tune_models(X_norm_train, y_train, hyperoptimizer, split_idx=i, progress=progress)
            else:
                new_models = deepcopy(self.models)

            # Evaluate models
            model_tuples = self._evaluate_models(X_norm_train, y_train, X_norm_test, y_test, new_models, split_idx=i, progress=progress)
            _model_names, trained_models, _results = zip(*model_tuples)

            # Add results to dataframe
            for model_name, _trained_model, res in model_tuples:
                results_val_train, results_val_test, result_train, result_test = res
                for res_val_train, res_val_test in zip(results_val_train, results_val_test):
                    val_seed = res_val_train[0]
                    cv_fold = res_val_train[1]

                    val = {"Model": model_name, "Train Seed": seed, "Validation Seed": val_seed, "CV Fold": cv_fold}
                    for i, (metric_name, _, _) in enumerate(self.metrics):
                        val[f"{metric_name} / Train"] = result_train[i]
                        val[f"{metric_name} / Test"] = result_test[i]
                        val[f"{metric_name} / Validation Train"] = res_val_train[2][i]
                        val[f"{metric_name} / Validation Test"] = res_val_test[2][i]

                    report.append(val)

            # Add normalization to models with pipelines
            if self._input_transform:
                trained_models = [
                    Pipeline(
                        [
                            ("scaler", self._input_transform),
                            ("model", model),
                        ]
                    )
                    for model in trained_models
                ]

            if progress is not None:
                progress.update(seed_progress, advance=1)

        if progress is not None:
            progress.update(seed_progress, advance=1)

        # Create dataframe
        full_report = pd.DataFrame(report)

        res_cols = [f"{m} / {v}" for (m, _, _) in self.metrics for v in ["Validation Train", "Validation Test", "Train", "Test"]]
        report = full_report.groupby(["Model"]).agg({col: ["mean", "std"] for col in res_cols}).reset_index()
        report.columns = report.columns.to_flat_index().str.join(" / ")
        report.columns = report.columns.str.replace("mean", "Mean").str.replace("std", "Std")
        report.columns = report.columns.str.replace("Model / ", "Model")

        # Store data in the object's attributes
        self._report = report
        self._full_report = full_report
        self._trained_models = [(m_name, tmodel, param_grid) for (m_name, _, param_grid), tmodel in zip(self.models, trained_models)]

        return report

    def tune_models(
        self, X: np.ndarray, y: np.ndarray, hyperoptimizer: HyperOptimizer = None, split_idx: int = 1, progress: rich.progress.Progress = None
    ) -> List[Tuple[str, BaseEstimator, dict]]:
        """
        Tune a model using cross-validation.

        :param X: Input observations.
        :type X: ndarray
        :param y: Target values.
        :type y: ndarray
        :param hyperoptimizer: Hyperparameter optimizer that will find the best parameters for each model.
        :type hyperoptimizer: HyperOptimizer
        :return: The tuned model.
        :rtype: List[Tuple[str, BaseEstimator, dict]]
        """

        best_param_list = self.optimize_hyperparams(X, y, hyperoptimizer, split_idx=split_idx, progress=progress)

        new_models = []
        for (model_name, model, model_param_grid), best_params in zip(self.models, best_param_list):
            # change the parameters without overwriting the model
            configured_model = copy(model).set_params(**best_params)

            new_models.append((model_name, configured_model, model_param_grid))

        return new_models

    def optimize_hyperparams(
        self, X: np.ndarray, y: np.ndarray, hyperoptimizer: HyperOptimizer = None, split_idx: int = 1, progress: rich.progress.Progress = None
    ) -> List[dict]:
        """
        Obtain the best set of parameters for each of the models.

        :param X: Input data.
        :type X: ndarray
        :param y: Target data.
        :type y: ndarray
        :param hyperoptimizer: Hyperparameter optimizer that will find the best parameters for each model.
        :type hyperoptimizer: HyperOptimizer
        :return: List of the best hyperparameters obtained for each model.
        :rtype: List[dict]
        """

        self._best_params = []

        if progress is not None:
            optimizer_progress = progress.add_task(f"[blue]  Optimizing models [Split {split_idx+1}]", total=len(self.models))

        for name, model, param_grid in self.models:
            best_params_model = hyperoptimizer.best_params(X, y, model, param_grid)
            self._best_params.append(best_params_model)

            if progress is not None:
                padded_name = name.ljust(self.model_name_str_pad)
                if len(padded_name) > self.model_name_str_pad:
                    padded_name = padded_name[: self.model_name_str_pad - 3] + "..."
                progress.update(optimizer_progress, advance=1, description=f"[blue]  Optimizing models [Split {split_idx+1}]: {padded_name}")

        if progress is not None:
            progress.update(
                optimizer_progress,
                completed=len(self.models),
                description=f"[blue]  Optimizing models [Split {split_idx+1}] " + " " * self.model_name_str_pad,
            )

        return self._best_params

    def _evaluate_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Iterable[Tuple[str, BaseEstimator, dict]],
        split_idx: int = 1,
        progress: rich.progress.Progress = None,
    ) -> list[tuple[Any, Any, tuple[list[Any], list[Any], list[float], list[float]]]]:
        """
        Evaluate all models on a given dataset with their default hyperparameters.

        :param X_train: Input observations in the training set.
        :type X_train: ndarray
        :param y_train: Target values in the training set.
        :type y_train: ndarray
        :param X_test: Input observations in the test set.
        :type X_test: ndarray
        :param y_test: Target values in the test set.
        :type y_test: ndarray
        :return: A DataFrame with the results.
        :rtype: list[tuple[Any, Any, tuple[list[Any], list[Any], list[float], list[float]]]]
        """

        results = []

        if progress is not None:
            model_progress = progress.add_task(f"[blue]  Evaluating models [Split {split_idx+1}]", total=len(models))

        for i, (name, model, _) in enumerate(models):
            hh_logger.info(f"Running model [{i + 1}/{len(models)}]: {name}")
            res_val_train, res_val_test = self._evaluate_model_cv_multiple_seeds(X_train, y_train, model, n_evals=self.n_val_evals, progress=progress)

            trained_model = copy(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trained_model.fit(X_train, y_train)

            res_train = self._calc_metrics(y_train, trained_model.predict(X_train))
            res_test = self._calc_metrics(y_test, trained_model.predict(X_test))

            results.append((name, trained_model, (res_val_train, res_val_test, res_train, res_test)))

            if progress is not None:
                padded_name = name.ljust(self.model_name_str_pad)
                if len(padded_name) > self.model_name_str_pad:
                    padded_name = padded_name[: self.model_name_str_pad - 3] + "..."
                progress.update(model_progress, advance=1, description=f"[blue]  Evaluating models [Split {split_idx+1}]: {padded_name}")

        if progress is not None:
            progress.update(
                model_progress, completed=len(models), description=f"[blue]  Evaluating models [Split {split_idx+1}] " + " " * self.model_name_str_pad
            )

        return results

    def _evaluate_model_cv_multiple_seeds(
        self, X_train: np.ndarray, y_train: np.ndarray, model: BaseEstimator, n_evals: int = 10, progress: rich.progress.Progress = None
    ) -> tuple[list[Any], list[Any]]:
        """
        Evaluate a model multiple times, with a different seed every time.

        :param X_train: Input observations in the training set.
        :type X_train: ndarray
        :param y_train: Target values in the training set.
        :type y_train: ndarray
        :param model: Model to evaluate.
        :type model: BaseEstimator
        :param n_evals: Number of times to train the model (each iteration uses a different seed).
        :type n_evals: int
        :return: A tuple with the results for validation train, validation test, train and test.
        :rtype: tuple[list[Any], list[Any]]
        """

        results_val_train, results_val_test = [], []
        seeds = self._generate_seeds(n_evals, self.seed_strategy)

        for i, seed in enumerate(seeds):
            hh_logger.debug(f"Iteration [{i + 1}/{n_evals - 1}]")
            result_val_train, result_val_test = self._evaluate_model_cv(X_train, y_train, model, seed=seed)

            results_val_train += result_val_train
            results_val_test += result_val_test

        results = (results_val_train, results_val_test)

        model_info = "Metrics:\n"
        for i, (metric_name, _, _) in enumerate(self.metrics):
            model_info += f"{i}: {metric_name}\n"
            for j, data_name in enumerate(["Validation Train", "Validation Test"]):
                avg_res = np.mean([m[i] for m in results[j]])
                std_res = np.std([m[i] for m in results[j]])
                model_info += f"\t{data_name}: {avg_res:.3f} ± {std_res:.3f}\n"
            model_info += "\n"
        hh_logger.info(model_info)

        return results_val_train, results_val_test

    def _evaluate_model_cv(
        self, X_train: np.ndarray, y_train: np.ndarray, model: BaseEstimator, seed: int = 0
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Evaluate a model on a given dataset.

        :param X_train: Input observations in the training set.
        :type X_train: ndarray
        :param y_train: Target values in the training set.
        :type y_train: ndarray
        :param model: Model to evaluate.
        :type model: BaseEstimator
        :param seed: Random seed.
        :type seed: int
        :return: A tuple with the results for validation train, validation test, train and test.
        :rtype: tuple[list[list[float]], list[list[float]]]
        """

        if hasattr(model, "random_state"):
            model.random_state = seed

        cv = self._create_cross_validator(seed)

        results_val_train, results_val_test = [], []

        for split_idx, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
            val_model = copy(model)
            hh_logger.debug(f"Split [{split_idx}/{cv.get_n_splits(X_train, y_train)}]")

            X_val_train, X_val_test = X_train[train_index], X_train[test_index]
            y_val_train, y_val_test = y_train[train_index], y_train[test_index]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val_model.fit(X_val_train, y_val_train)

            result_val_train = self._calc_metrics(y_val_train, val_model.predict(X_val_train))
            result_val_test = self._calc_metrics(y_val_test, val_model.predict(X_val_test))

            results_val_train.append([seed, split_idx, result_val_train])
            results_val_test.append([seed, split_idx, result_val_test])

        return results_val_train, results_val_test

    def _create_cross_validator(self, seed):
        cv = self.cross_validator(**self.cross_validator_params)
        if hasattr(cv, "random_state"):
            cv.random_state = seed
        return cv

    def _stratify_array(self, _y):
        return None

    def _generate_seeds(self, n, seed_strategy):
        if seed_strategy == "random":
            return np.random.randint(0, 100000, n)
        elif seed_strategy == "sequential":
            return list(range(n))
        else:
            raise ValueError(f"Invalid seed strategy: {seed_strategy}")
