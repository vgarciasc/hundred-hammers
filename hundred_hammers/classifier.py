from copy import deepcopy, copy
from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_CLASSIFICATION_MODELS, DEFAULT_CLASSIFICATION_METRICS


class HundredHammersClassifier(HundredHammersBase):
    """
    HundredHammers class specialized in classification models.
    Implements methods for automatic machine learning like evaluating a list of models
    and performing hyperparameter optimization.

    :param models: List of models to evaluate (has a default list of models)
    :param metrics: Metrics to use to evaluate the models (has a default list of metrics)
    :param eval_metric: Target metric to use in hyperparameter optimization (default is the first metric in metrics)
    :param test_size: Percentage of the dataset to use for testing (default 0.2)
    :param n_folds: Number of Cross Validation folds (default 5)
    :param n_folds_tune: Number of Cross Validation folds in grid search (default 5)
    :param n_evals: Number of times to repeat the training of the models (default 10)
    :param show_progress_bar: Show progress bar in the evaluation (default False)
    :param seed_strategy: Strategy used to generate the seeds for the different evaluations ('sequential' or 'random')
    """

    def __init__(
        self,
        models=None,
        metrics=None,
        eval_metric=None,
        input_transform=None,
        test_size=0.2,
        n_folds=5,
        n_folds_tune=5,
        n_evals=10,
        show_progress_bar=False,
        seed_strategy='sequential'
    ):
        if models is None:
            models = deepcopy(DEFAULT_CLASSIFICATION_MODELS)

        if metrics is None:
            metrics = copy(DEFAULT_CLASSIFICATION_METRICS)

        super().__init__(
            models,
            metrics,
            eval_metric,
            input_transform,
            test_size,
            n_folds,
            n_folds_tune,
            n_evals,
            show_progress_bar,
            seed_strategy
        )
