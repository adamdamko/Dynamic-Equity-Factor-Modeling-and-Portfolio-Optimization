# This is the pipeline for data_processing

from kedro.pipeline import Pipeline, node
from .nodes import join_data, intro_clean_data, clean_data_eda


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # node(
            #     func=SimFinAPIDownload.pull_daily_prices,
            #     inputs=['params:simfin_api'],
            #     outputs='prices',
            #     name='prices_node',
            # ),
            # node(
            #     func=SimFinAPIDownload.pull_us_companies,
            #     inputs=['params:simfin_api'],
            #     outputs='companies',
            #     name='companies_node',
            # ),
            # node(
            #     func=SimFinAPIDownload.pull_industries,
            #     inputs=['params:simfin_api'],
            #     outputs='industries',
            #     name='industries_node',
            # ),
            # node(
            #     func=SimFinAPIDownload.pull_share_price_ratios,
            #     inputs=['params:simfin_api'],
            #     outputs='shareprice_ratios',
            #     name='shareprice_ratios_node',
            # ),
            node(
                func=join_data,
                inputs=['companies', 'industries', 'prices', 'ratios'],
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
