# This is the pipeline for models.

from kedro.pipeline import Pipeline, node
from .nodes import TrainTestValidation, HyperparameterTuning


def create_test_train_validation_sets_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=TrainTestValidation.holdout_set,
                inputs=['modeling_data'],
                outputs='holdout_data',
                name='holdout_data_node',
            ),
            node(
                func=TrainTestValidation.train_val_x,
                inputs=['modeling_data'],
                outputs='train_x_data',
                name='train_x_data_node',
            ),
            node(
                func=TrainTestValidation.train_val_y,
                inputs=['modeling_data'],
                outputs='train_y_data',
                name='train_y_data_node',
            ),
            node(
                func=TrainTestValidation.time_series_split,
                inputs=['modeling_data'],
                outputs='time_series_split_list',
                name='time_series_split_list_node',
            ),
        ]
    )


def create_hyperparameter_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=HyperparameterTuning.train_cv_lgbm,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:lgbm_static_params',
                        'params:lgbm_fine_tune_params'],
                outputs='lgbm_fine_cv_model',
                name='lgbm_fine_cv_model_node',
            ),
        ]
    )
