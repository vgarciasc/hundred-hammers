from copy import copy, deepcopy

from sklearn.model_selection import KFold

from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_REGRESSION_MODELS, DEFAULT_REGRESSION_METRICS


class HundredHammersRegressor(HundredHammersBase):
    """
    HundredHammers class specialized in regression models.
    Implements methods for automatic machine learning like evaluating a list of models
    and performing hyperparameter optimization.

    :param models: List of models to evaluate (has a default list of models)
    :param metrics: Metrics to use to evaluate the models (has a default list of metrics)
    :param eval_metric: Target metric to use in hyperparameter optimization (default is the first metric in metrics)
    :param input_transform: Input normalization strategy used. Specified as a string or the normalization class. ('MinMax', 'MaxAbs', 'Standard', 'Norm', 'Robust')
    :param cross_validator: Cross Validator to use in the evaluation (default KFold)
    :param cross_validator_params: Parameters for the Cross Validator (default {"shuffle": True, "n_splits": 5})
    :param test_size: Percentage of the dataset to use for testing (default 0.2)
    :param n_train_evals: Number of times to vary the training/test separation seed.
    :param n_val_evals: Number of times to vary the cross-validation seed.
    :param show_progress_bar: Show progress bar in the evaluation (default False)
    :param seed_strategy: Strategy used to generate the seeds for the different evaluations ('sequential' or 'random')
    """

    def __init__(
        self,
        models=None,
        metrics=None,
        eval_metric=None,
        input_transform=None,
        cross_validator=KFold,
        cross_validator_params=None,
        test_size=0.2,
        n_val_evals=1,
        n_train_evals=1,
        show_progress_bar=False,
        seed_strategy="sequential",
    ):
        if models is None:
            models = deepcopy(DEFAULT_REGRESSION_MODELS)

        if metrics is None:
            metrics = copy(DEFAULT_REGRESSION_METRICS)

        if cross_validator_params is None:
            cross_validator_params = {"shuffle": True, "n_splits": 5}

        super().__init__(
            models=models,
            metrics=metrics,
            eval_metric=eval_metric,
            input_transform=input_transform,
            cross_validator=cross_validator,
            cross_validator_params=cross_validator_params,
            test_size=test_size,
            n_train_evals=n_train_evals,
            n_val_evals=n_val_evals,
            show_progress_bar=show_progress_bar,
            seed_strategy=seed_strategy,
        )
