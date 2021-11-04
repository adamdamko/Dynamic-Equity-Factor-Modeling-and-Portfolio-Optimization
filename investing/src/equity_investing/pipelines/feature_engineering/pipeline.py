# This is the pipeline for feature_engineering

from kedro.pipeline import Pipeline, node
from typing import Dict
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
                inputs=['filtered_data', 'params:current_date'],
                outputs='filtered_dates_data',
                name='filtered_dates_data_node',
            ),
            node(
                func=FeatureEngineering.create_returns,
                inputs=['filtered_dates_data', 'params:change_index_list'],
                outputs='returns_data',
                name='returns_data_node',
            ),
            node(
                func=FeatureEngineering.create_rolling_values,
                inputs=['returns_data', 'params:sum_list', 'params:avg_list', 'params:med30_list'],
                outputs='rolling_values_data',
                name='rolling_values_data_node',
            ),
            node(
                func=FeatureEngineering.market_schedule,
                inputs=['params:current_date'],
                outputs='market_schedule_data',
                name='market_schedule_node',
            ),
            node(
                func=FeatureEngineering.lagged_rolling_returns,
                inputs=['rolling_values_data', 'market_schedule_data', 'params:sum_list', 'params:avg_list'],
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
                inputs=['monthly_data', 'params:momentum_features', 'params:mom_1m_numerators_list',
                        'params:mom_numerators_list', 'params:mom_1m_denominators_list',
                        'params:mom_3m_denominators_list', 'params:mom_6m_denominators_list',
                        'params:mom_12m_denominators_list', 'params:mom_feature_names_list', 'params:lag_range'],
                outputs='momentum_data',
                name='create_momentum_factors_node',
            ),
            node(
                func=FeatureEngineering.create_modeling_data,
                inputs=['momentum_data', 'params:modeling_data_drop_list', 'params:model_target'],
                outputs='modeling_data',
                name='create_modeling_data_node',
            ),
        ]
    )


def create_final_eda_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=FeatureEngineering.create_final_eda_data,
                inputs=['modeling_data'],
                outputs='eda_modeling_data',
                name='eda_modeling_data_node',
            ),
        ]
    )
