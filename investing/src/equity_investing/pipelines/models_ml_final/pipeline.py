# This is the pipeline for models_ml_final.

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


def create_coarse_hyperparameter_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=HyperparameterTuning.train_cv_lgbm,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:lgbm_static_params',
                        'params:lgbm_coarse_tune_params', 'params:lgbm_features'],
                outputs='lgbm_coarse_cv_model',
                name='lgbm_coarse_cv_model_node',
            ),
            node(
                func=HyperparameterTuning.visualize_hyperparameters,
                inputs=['lgbm_coarse_cv_model'],
                outputs='lgbm_coarse_cv_model_hyperparameters',
                name='lgbm_coarse_cv_model_hyperparameters_node',
            ),
        ]
    )


def create_fine_hyperparameter_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=HyperparameterTuning.train_cv_lgbm,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:lgbm_static_params',
                        'params:lgbm_fine_tune_params', 'params:lgbm_features'],
                outputs='lgbm_fine_cv_model',
                name='lgbm_fine_cv_model_node',
            ),
            node(
                func=HyperparameterTuning.visualize_hyperparameters,
                inputs=['lgbm_fine_cv_model'],
                outputs='lgbm_fine_cv_model_hyperparameters',
                name='lgbm_fine_cv_model_hyperparameters_node',
            ),
            node(
                func=HyperparameterTuning.random_state_test,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:lgbm_static_params',
                        'params:lgbm_fine_best_params', 'params:lgbm_features'],
                outputs='lgbm_fine_random_state_test',
                name='lgbm_fine_random_state_test',
            ),
        ]
    )
