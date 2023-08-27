from hundred_hammers.base import HundredHammersBase
from hundred_hammers.model_zoo import DEFAULT_CLASSIFICATION_MODELS, DEFAULT_CLASSIFICATION_METRICS


class HundredHammersClassifier(HundredHammersBase):
    def __init__(self, models=DEFAULT_CLASSIFICATION_MODELS, metrics=DEFAULT_CLASSIFICATION_METRICS,
                 test_size=0.2, n_folds=5, n_folds_tune=5, n_seeds=10, verbose=True):
        super().__init__(models, metrics, test_size, n_folds, n_folds_tune, n_seeds, verbose)