# This is the pipeline for portfolio_backtesting.

from kedro.pipeline import Pipeline, node
from .nodes import PortfolioBacktesting


def portfolio_backtesting_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=PortfolioBacktesting.market_dates,
                inputs=['params:schedule_end_date', 'params:last_prediction_date'],
                outputs='market_dates_data',
                name='market_dates_data_node',
            ),
            node(
                func=PortfolioBacktesting.portfolio_backtesting,
                inputs=['top_predictions_view_data', 'intro_cleaned_data', 'market_dates_data',
                        'params:initial_portfolio_value', 'params:confidence_level'],
                outputs='backtesting_results_data',
                name='backtesting_results_data_node',
            ),
        ]
    )
