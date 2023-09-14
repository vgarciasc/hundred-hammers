from copy import deepcopy, copy
from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_CLASSIFICATION_MODELS, DEFAULT_CLASSIFICATION_METRICS


class HundredHammersClassifier(HundredHammersBase):
    """
    HundredHammers class specialized on classification models.
    """

    def __init__(self, models=None, metrics=None, eval_metric = None, test_size=0.2,
                 n_folds=5, n_folds_tune=5, n_seeds=10):
        if models is None:
            models = deepcopy(DEFAULT_CLASSIFICATION_MODELS)

        if metrics is None:
            metrics = copy(DEFAULT_CLASSIFICATION_METRICS)

        super().__init__(models, metrics, eval_metric, test_size, n_folds, n_folds_tune, n_seeds)
