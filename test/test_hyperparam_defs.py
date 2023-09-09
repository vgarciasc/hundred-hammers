# pylint: skip-file

import pytest
from hundred_hammers import known_hyperparams, known_models, hyperparam_def_schema, add_known_model_def
from schema import SchemaError

def test_verify_predef_params():
    for known_model_hyperparams in known_hyperparams:
        hyperparam_def_schema.validate(known_model_hyperparams)

@pytest.mark.parametrize("new_hyperparams", [
    {'model': 'Ex1', 'param1': {'type': 'real', 'min': 0, 'max': 10.24}},
    {'model': 'Ex2', 'param1': {'type': 'integer', 'min': 0, 'max': 5}},
    {'model': 'Ex3', 'param1': {'type': 'categorical', 'values': ["manhattan", "euclidean"]}},
    {'model': 'Ex4', 'param1': {'type': 'categorical', 'values': [0, 2, 6]}},
    {'model': 'Ex5', 'param1': {'type': 'real', 'min': 0, 'max': 10.24},
                     'param2': {'type': 'categorical', 'values': ["manhattan", "euclidean"]}},
    {'model': 'Ex6', 'param1': {'type': 'integer', 'min': 0, 'max': 10},
                     'param2': {'type': 'real', 'min': 0, 'max': 10.24},
                     'param3': {'type': 'categorical', 'values': ["manhattan", "euclidean"]}},
    {'model': 'Ex7'}
])
def test_add_to_known_hyperparams(new_hyperparams):
    add_known_model_def(new_hyperparams)
    assert new_hyperparams in known_hyperparams
    assert new_hyperparams['model'] in known_models

@pytest.mark.parametrize("new_hyperparams", [
    {'model': 'Bad1', 'param1': {'type': 'real', 'min': 0}},
    {'model': 'Bad1', 'param1': {'type': 'real', 'max': 0}},
    {'model': 'Bad2', 'param1': {'type': 'integer'}},
    {'model': 'Bad3', 'param1': {'type': 'categorical'}},
    {'model': 'Bad4', 'param1': {'type': 'categorical', 'min': 0, 'max': 5}},
    {'param1': {'type': 'categorical', 'values': [0, 2, 6]}}
])
def test_add_to_known_hyperparams_bad(new_hyperparams):
    with pytest.raises(SchemaError):
        add_known_model_def(new_hyperparams)
    assert new_hyperparams not in known_hyperparams
    assert new_hyperparams['model'] not in known_models if 'model' in new_hyperparams else True
