from copy import copy, deepcopy
from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_REGRESSION_MODELS, DEFAULT_REGRESSION_METRICS


class HundredHammersRegressor(HundredHammersBase):
    """
    HundredHammers class specialized in regression models.
    """

    def __init__(self, models=None, metrics=None, eval_metric=None, test_size=0.2,
                 n_folds=5, n_folds_tune=5, n_seeds=10, seed_strategy='sequential'):
        if models is None:
            models = deepcopy(DEFAULT_REGRESSION_MODELS)

        if metrics is None:
            metrics = copy(DEFAULT_REGRESSION_METRICS)

        super().__init__(models, metrics, eval_metric, test_size, n_folds, n_folds_tune, n_seeds, seed_strategy)
