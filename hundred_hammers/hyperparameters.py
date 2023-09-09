from __future__ import annotations
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import warnings
from .config import hh_logger
import random

from copy import deepcopy
from pathlib import Path
import json

# Read the predefined hyperpameter definitions
_base_path = Path(__file__).parents[0]
_hp_filepath = next(_base_path.glob("hyperparam_info.json"))

with open(_hp_filepath, "r") as fjson:
    _known_hyperparams = json.load(fjson)
    _known_models = [k["model"] for k in _known_hyperparams]


def find_hyperparam_grid(model: BaseEstimator, n_grid_points: int = 10):
    """
    Obtains a grid of hyperparameters to optimize for the model.

    :param model: List of model for which we want to find the hyperparameters.
    :return: List with the hyperparameter definition for each of the model passed
    """

    hyperparam_def = find_hyperparam_def(model)
    if hyperparam_def:
        hyperparam_grid = construct_hyperparam_grid(hyperparam_def, n_grid_points)
    else:
        hyperparam_grid = {}

    return hyperparam_grid

def find_hyperparam_def(model: BaseEstimator) -> dict:
    """
    Obtains the definitions of the hyperparameters of each of the model listed.

    :param model: List of model for which we want to find the hyperparameters.
    :return: List with the hyperparameter definition for each of the model passed
    """

    model_name = type(model).__name__
    params = None

    if model_name in _known_models:
        p_idx = _known_models.index(model_name)
        params = _known_hyperparams[p_idx]
        params.pop("model", None)
    else:
        hh_logger.warn(f"The model {model_name} has not been found and no hyperparameter definitions can be used.")
    
    return params

def construct_hyperparam_grid(hyperparam_grid_def: dict, n_grid_points: int = 10) -> dict:
    """
    Generate a grid of hyperparameters from their definition.

    :param hyperparam_grid_def: Definition of the hyperparameters to be generated as a grid.
    :param n_grid_points: Number of values to pick for each hyperparameter.
    :return: List of hyperparameter grids to use in grid search.
    """
    
    keys = list(hyperparam_grid_def.keys())
    if "model" in keys:
        keys.remove("model")

    model_params = {}
    hh_logger.debug(f"Using hyperparameter definitions: {hyperparam_grid_def}")
    for k in keys:
        if hyperparam_grid_def[k]["type"] == "integer":
            model_params[k] = np.linspace(hyperparam_grid_def[k]["min"], hyperparam_grid_def[k]["max"], n_grid_points)
            model_params[k] = np.unique(np.round(model_params[k])).astype(int)
        
        elif hyperparam_grid_def[k]["type"] == "real":
            model_params[k] = np.geomspace(max(hyperparam_grid_def[k]["min"], 1e-10), hyperparam_grid_def[k]["max"], n_grid_points)
        
        elif hyperparam_grid_def[k]["type"] == "categorical":
            values = hyperparam_grid_def[k]["values"]
            model_params[k] = random.sample(values, k=min(len(values), n_grid_points)) # take at most n random non-repeating values
            model_params[k] = [i if i != 'None' else None for i in model_params[k]] # Interpret 'None' as a python null value

    return model_params