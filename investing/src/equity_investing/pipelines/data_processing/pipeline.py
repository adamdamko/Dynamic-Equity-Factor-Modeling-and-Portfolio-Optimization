# This is the pipeline for data_processing

from kedro.pipeline import Pipeline, node
from .nodes import join_data, intro_clean_data, clean_data_eda


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=join_data,
                inputs=['companies', 'industries', 'prices'],
                outputs='raw_joined_data',
                name='join_data_node',
            ),
            node(
                func=intro_clean_data,
                inputs=['raw_joined_data'],
                outputs='intro_cleaned_data',
                name='intro_clean_data_node',
            ),
        ]
    )


def create_eda_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=clean_data_eda,
                inputs=['intro_cleaned_data'],
                outputs='cleaned_data_eda',
                name='cleaned_data_eda_node',
            ),
        ]
    )
