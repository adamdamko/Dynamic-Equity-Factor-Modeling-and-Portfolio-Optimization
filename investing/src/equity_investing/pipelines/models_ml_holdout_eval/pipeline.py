# This is the pipeline for models_ml_holdout_eval.

from kedro.pipeline import Pipeline, node
from .nodes import HoldoutValidation


def create_holdout_validation_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=HoldoutValidation.validation,
                inputs=['modeling_data', 'params:validation_train_dates_list', 'params:validation_test_dates_list',
                        'params:lgbm_fine_best_params', 'params:model_features', 'params:model_target'],
                outputs='holdout_results_data',
                name='holdout_results_data_node',
            ),
        ]
    )
