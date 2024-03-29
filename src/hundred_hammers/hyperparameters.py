from __future__ import annotations
from copy import copy
import random
import json
from pathlib import Path
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from .config import hh_logger
from .hyperparam_info import known_hyperparams, known_models, hyperparam_def_schema


def add_known_model_def(def_dict: dict):
    """
    Adds the definition of the hyperparameters of a new model to the
    list of know hyperparameters and known models.

    The definition should be a dictionary that follows this schema: ::

        {
            'model': <Name>,
            <hyperparam_name>:
                Choose one of:
                - {"type": "real", "min": <number>, "max": <number>},
                - {"type": "integer", "min": <number>, "max": <number>},
                - {"type": "categorical", "values": [<any>]}
        }

    There can be any number of hyperparameters, even 0, they MUST correspond to the
    arguments used in the model constructor, or you will get an error in the hyperparameter
    search step.

    :param def_dict: dictionary that defines the hyperparameters of a new model.
    :type def_dict: dict
    """

    if hyperparam_def_schema.is_valid(def_dict):
        known_hyperparams.append(def_dict)
        known_models.append(def_dict["model"])
    else:
        hh_logger.error("The hyperparameter definition provided has an incorrect format.")

        # Since the schema is incorrect, validating the schema with raise an error with
        # detailed information about why it failed.
        hyperparam_def_schema.validate(def_dict)


def find_hyperparam_def(model: BaseEstimator) -> dict:
    """
    Obtains the definitions of the hyperparameters of each of the model listed.

    :param model: Model for which we want to find the hyperparameters.
    :type model: BaseEstimator
    :return: Hyperparameter definition for the model.
    :rtype: dict
    """

    model_name = type(model).__name__
    params = None

    if model_name in known_models:
        p_idx = known_models.index(model_name)
        params = copy(known_hyperparams[p_idx])
        params.pop("model", None)
    else:
        hh_logger.warning(f"The model {model_name} has not been found and no hyperparameter definitions can be used.")

    return params


def find_hyperparam_grid(model: BaseEstimator, n_grid_points: int = 10) -> dict:
    """
    Obtains a grid of hyperparameters to optimize for the model.

    :param model: Model for which we want to find the hyperparameters.
    :type model: BaseEstimator
    :param n_grid_points: Number of values to pick for each hyperparameter.
    :type n_grid_points: int
    :return: Hyperparameter definition for the model.
    :rtype: dict
    """

    hyperparam_def = find_hyperparam_def(model)
    if hyperparam_def:
        hyperparam_grid = construct_hyperparam_grid(hyperparam_def, n_grid_points)
        hh_logger.info(f"The hyperparameter grid for the model {type(model).__name__} was generated succesfully.")
    else:
        hyperparam_grid = {}

    return hyperparam_grid


def construct_hyperparam_grid(hyperparam_grid_def: dict, n_grid_points: int = 10) -> dict:
    """
    Generate a grid of hyperparameters from their definition.

    :param hyperparam_grid_def: Definition of the hyperparameters to be generated as a grid.
    :type hyperparam_grid_def: dict
    :param n_grid_points: Number of values to pick for each hyperparameter.
    :type n_grid_points: int
    :return: Hyperparameter grid to use in grid search.
    :rtype: dict
    """

    keys = [k for k in hyperparam_grid_def.keys() if k != "model"]

    model_params = {}
    hh_logger.debug(f"Using hyperparameter definitions: {hyperparam_grid_def}")
    for k in keys:
        if hyperparam_grid_def[k]["type"] == "integer":
            model_params[k] = np.linspace(hyperparam_grid_def[k]["min"], hyperparam_grid_def[k]["max"], n_grid_points)

            # Convert linspace generated numbers to integers
            model_params[k] = np.unique(np.round(model_params[k])).astype(int)

        elif hyperparam_grid_def[k]["type"] == "real":
            model_params[k] = np.geomspace(max(hyperparam_grid_def[k]["min"], 1e-10), hyperparam_grid_def[k]["max"], n_grid_points)

        elif hyperparam_grid_def[k]["type"] == "categorical":
            values = hyperparam_grid_def[k]["values"]

            # Take at most n random non-repeating values
            model_params[k] = random.sample(values, k=min(len(values), n_grid_points))

            # Interpret 'None' as a python null value
            model_params[k] = [i if i != "None" else None for i in model_params[k]]

    return model_params


def find_hyperparam_random(model: BaseEstimator, n_samples: int = 10) -> dict:
    """
    Obtains a grid of hyperparameters to optimize for the model.

    :param model: Model for which we want to find the hyperparameters.
    :type model: BaseEstimator
    :param n_grid_points: Number of values to pick for each hyperparameter.
    :type n_grid_points: int
    :return: Hyperparameter definition for the model.
    :rtype: dict
    """

    hyperparam_def = find_hyperparam_def(model)
    if hyperparam_def:
        hyperparam_grid = construct_hyperparam_random(hyperparam_def, n_samples)
        hh_logger.info(f"The hyperparameter grid for the model {type(model).__name__} was generated succesfully.")
    else:
        hyperparam_grid = {}

    return hyperparam_grid


def construct_hyperparam_random(hyperparam_grid_def: dict, n_samples: int = 10) -> dict:
    """
    Generate a grid of hyperparameters from their definition.

    :param hyperparam_grid_def: Definition of the hyperparameters to be generated as a grid.
    :type hyperparam_grid_def: dict
    :param n_grid_points: Number of values to pick for each hyperparameter.
    :type n_grid_points: int
    :return: List of hyperparameter grids to use in grid search.
    :rtype: dict
    """

    keys = [k for k in hyperparam_grid_def.keys() if k != "model"]

    model_params = {}
    hh_logger.debug(f"Using hyperparameter definitions: {hyperparam_grid_def}")
    for k in keys:
        if hyperparam_grid_def[k]["type"] == "integer":
            model_params[k] = sp.stats.distributions.randint(hyperparam_grid_def[k]["min"], hyperparam_grid_def[k]["max"])

        elif hyperparam_grid_def[k]["type"] == "real":
            model_params[k] = sp.stats.distributions.loguniform(max(hyperparam_grid_def[k]["min"], 1e-10), hyperparam_grid_def[k]["max"])

        elif hyperparam_grid_def[k]["type"] == "categorical":
            # Interpret 'None' as a python null value
            model_params[k] = [i if i != "None" else None for i in hyperparam_grid_def[k]["values"]]

    return model_params
