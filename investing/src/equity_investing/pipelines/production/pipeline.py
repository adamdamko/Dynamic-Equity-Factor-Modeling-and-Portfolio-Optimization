# This is the pipeline for production.

from kedro.pipeline import Pipeline, node
from .nodes import Production


def production_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=Production.production_model,
                inputs=['modeling_data', 'params:production_train_date', 'params:production_test_date',
                        'params:lgbm_fine_best_params', 'params:model_features', 'params:model_target'],
                outputs='production_predictions_data',
                name='production_predictions_data_node',
            ),
            node(
                func=Production.market_production_dates,
                inputs=['params:production_start_date', 'params:production_end_date', 'params:last_production_date'],
                outputs='production_dates_data',
                name='production_dates_data_node',
            ),
            node(
                func=Production.portfolio_optimization,
                inputs=['production_predictions_data', 'intro_cleaned_data', 'production_dates_data',
                        'params:confidence_level'],
                outputs='production_results_data',
                name='production_results_data_node',
            ),
        ]
    )
