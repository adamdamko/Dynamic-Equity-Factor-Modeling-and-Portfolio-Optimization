# This is the pipeline for models_ml_exploratory.

from kedro.pipeline import Pipeline, node
from .nodes import ExploratoryModels


def create_exploratory_models_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=ExploratoryModels.train_cv_rf,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:rfr_static_params',
                        'params:rfr_tune_params'],
                outputs='rfr_model',
                name='rfr_model_node',
            ),
            node(
                func=ExploratoryModels.train_cv_xgb,
                inputs=['train_x_data', 'train_y_data', 'time_series_split_list', 'params:xgb_static_params',
                        'params:xgb_tune_params'],
                outputs='rfr_model',
                name='rfr_model_node',
            ),
        ]
    )
