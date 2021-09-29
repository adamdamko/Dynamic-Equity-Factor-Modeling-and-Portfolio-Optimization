# This is the pipeline for feature_engineering

from kedro.pipeline import Pipeline, node
from .nodes import DataFiltering, FeatureEngineering


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=FeatureEngineering.market_cap,
                inputs=['intro_cleaned_data'],
                outputs='market_cap_data',
                name='market_cap_data_node',
            ),
            node(
                func=DataFiltering.filter_data,
                inputs=['market_cap_data'],
                outputs='filtered_data',
                name='filtered_data_node',
            ),
            node(
                func=DataFiltering.filter_dates,
                inputs=['filtered_data'],
                outputs='filtered_dates_data',
                name='filtered_dates_data_node',
            ),
            node(
                func=FeatureEngineering.create_returns,
                inputs=['filtered_dates_data'],
                outputs='returns_data',
                name='returns_data_node',
            ),
            node(
                func=FeatureEngineering.create_rolling_values,
                inputs=['returns_data'],
                outputs='rolling_values_data',
                name='rolling_values_data_node',
            ),
            node(
                func=FeatureEngineering.market_schedule,
                inputs=[],
                outputs='market_schedule_data',
                name='market_schedule_node',
            ),
            node(
                func=FeatureEngineering.lagged_rolling_returns,
                inputs=['rolling_values_data', 'market_schedule_data'],
                outputs='lagged_rolling_returns_data',
                name='lagged_rolling_returns_node',
            ),
            node(
                func=FeatureEngineering.get_monthly,
                inputs=['lagged_rolling_returns_data'],
                outputs='monthly_data',
                name='get_monthly_node',
            ),
            node(
                func=FeatureEngineering.create_momentum_factors,
                inputs=['monthly_data'],
                outputs='momentum_data',
                name='create_momentum_factors_node',
            ),
            node(
                func=FeatureEngineering.create_modeling_data,
                inputs=['momentum_data'],
                outputs='modeling_data',
                name='create_modeling_data_node',
            ),
        ]
    )
