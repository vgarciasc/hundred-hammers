from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_REGRESSION_MODELS, DEFAULT_REGRESSION_METRICS


class HundredHammersRegressor(HundredHammersBase):
    """
    HundredHammers class specialized on regression models.
    """

    def __init__(self, models=DEFAULT_REGRESSION_MODELS, metrics=DEFAULT_REGRESSION_METRICS,
                 eval_metric=None, test_size=0.2, n_folds=5, n_folds_tune=5, n_seeds=10, verbose=True):
        super().__init__(models, metrics, eval_metric, test_size, n_folds, n_folds_tune, n_seeds, verbose)
