# This is the portfolio_backtesting nodes file.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import pypfopt
import pandas_market_calendars as mcal
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions


class PortfolioBacktesting:
    """
    The PortfolioBacktesting class takes predictions and historical prices. It performs
    Black-Litterman portfolio optimization per period. It then backtests the performance of
    these optimized portfolios over time.

    """
    @staticmethod
    def market_dates(schedule_end_date: Dict, last_prediction_date: Dict) -> pd.DataFrame:
        """
        This function creates the schedule of market dates to be used in portfolio optimization
        and backtesting.

        Args:
            schedule_end_date: The last date of the market schedule. This date should be
                               three months beyond the last prediction date. It should be an
                               end-of-calendar-month value. (E.g., if the last prediction date
                               is 2021-09-30 then the schedule_end_date would be 2021-12-31).
            last_prediction_date: Date of last prediction. (E.g., if you are predicting for the
                               month of 2021-10 ending 2021-10-29, the last_prediction_date would be
                               the last trading day of the prior month, 2021-09-30 in this case).

        Returns:
             Pandas dataframe with a schedule of dates to be used in portfolio optimization
             and portfolio backtesting.
        """
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        # Get date schedule
        schedule = nyse.schedule(start_date='2020-02-28', end_date=schedule_end_date)
        # Get month end
        schedule['year'] = schedule['market_close'].dt.year
        schedule['month'] = schedule['market_close'].dt.month
        schedule['day'] = schedule['market_close'].dt.day
        # Get number of days in group
        schedule['cal_days_in_month'] = schedule.groupby(['year', 'month'])['day'].transform('count')
        # Grab month end dates
        schedule_end = schedule.groupby(['year', 'month'])['day'].max().reset_index()
        schedule_end.loc[:, 'month_end_date'] = pd.to_datetime(schedule_end[['year', 'month', 'day']])
        # Grab month start dates
        schedule_start = schedule.groupby(['year', 'month'])['day'].min().reset_index()
        schedule_start.loc[:, 'month_start_date'] = pd.to_datetime(schedule_start[['year', 'month', 'day']])
        # Make schedule to join
        schedule_join = pd.merge(schedule_start, schedule_end, how='left', on=['year', 'month'], validate='1:1')
        schedule_join = schedule_join[['month_start_date', 'month_end_date']]
        # Create month ahead dates
        schedule_join.loc[:, 'period_start_date'] = schedule_join['month_start_date'].shift(-1)
        schedule_join.loc[:, 'period_end_date'] = schedule_join['month_start_date'].shift(-2)
        # Filter month_end_date to last prediction date
        schedule_join = schedule_join[schedule_join['month_end_date'] <= last_prediction_date]

        return schedule_join

    @staticmethod
    def portfolio_backtesting(top_predictions_view_data: pd.DataFrame, intro_cleaned_data: pd.DataFrame,
                              market_dates_data: pd.DataFrame, initial_portfolio_value: Dict,
                              confidence_level: Dict) -> pd.DataFrame:
        """
        This function performs portfolio optimization and backtesting.

        Args:
            top_predictions_view_data: Dataframe with top predictions from model eval.
            intro_cleaned_data: Initial cleaned data from data processing.
            market_dates_data: Schedule of dates to be used in portfolio optimization
                               and portfolio backtesting.
            initial_portfolio_value: Initial value to start portfolio backtesting.
            confidence_level: Confidence level in Black-Litterman optimization (note:
                              this could be different for all stock predictions, but
                              as this is algorithmic, I do not have any reason to have more/less
                               confidence in any given prediction).

        Returns:
             Pandas dataframe with the results from portfolio optimization and backtesting.
        """
        # Set initial parameters
        results = pd.DataFrame()
        initial_amount = initial_portfolio_value
        confidence = confidence_level

        # PORTFOLIO OPTIMIZATION AND BACKTESTING LOOP
        for date in list(market_dates_data['month_end_date']):
            # Get predictions by month
            preds_data = top_predictions_view_data[(top_predictions_view_data['date'] == date)]
            # Get price data used for risk model
            prices = (intro_cleaned_data[(intro_cleaned_data['date'] <= date) &
                                         (intro_cleaned_data['ticker'].isin(list(preds_data['ticker'])))])
            prices = prices[['date', 'ticker', 'market_cap', 'open', 'adj_close']]
            # Make time series of predictions
            prices_2 = prices.pivot(index='date', columns='ticker', values='adj_close').dropna()
            # Make risk model
            S = risk_models.CovarianceShrinkage(prices_2).ledoit_wolf()

            # BLACK-LITTERMAN PORTFOLIO OPTIMIZATION AND ALLOCATION
            # Get views and confidences
            predictions = preds_data.set_index('date')
            # Annualized returns for allocation
            predictions.loc[:, 'views'] = (predictions['predictions']) ** 12 - 1
            # Set confidence level
            predictions.loc[:, 'confidences'] = confidence
            # Make dict for views
            viewdict = predictions[['ticker', 'views']].reset_index(drop=True).set_index('ticker').to_dict()

            # Make Black-Litterman
            bl = BlackLittermanModel(cov_matrix=S,
                                     absolute_views=viewdict['views'],
                                     omega="idzorek",
                                     view_confidences=predictions['confidences']
                                     )
            # Posterior estimate of returns
            ret_bl = bl.bl_returns()
            S_bl = bl.bl_cov()
            # Portfolio allocation
            ef = EfficientFrontier(ret_bl, S_bl)
            ef.add_objective(objective_functions.L2_reg)
            ef.max_sharpe()
            weights = ef.clean_weights()
            # Make weights as dataframe
            weights = pd.merge(pd.DataFrame(weights.keys()), pd.DataFrame(weights.values()),
                               how='left',
                               left_index=True,
                               right_index=True).rename(columns={'0_x': 'ticker', '0_y': 'weights'})

            # CALCULATE PORTFOLIO PERFORMANCE
            # Join weights to date and price data
            results_df = pd.merge(weights, preds_data, how='left', left_on='ticker', right_on='ticker')
            # Merge schedule with results
            results_df2 = pd.merge(results_df, market_dates_data,
                                   how='left',
                                   left_on='date',
                                   right_on='month_end_date',
                                   validate='m:1')

            # Merge with prices for month ahead dates
            results_df3 = pd.merge(results_df2, intro_cleaned_data, how='left', left_on=['ticker', 'period_start_date'],
                                   right_on=['ticker', 'date'])
            results_df3 = pd.merge(results_df3,
                                   intro_cleaned_data,
                                   how='left',
                                   left_on=['ticker', 'simfinid', 'comp_name', 'period_end_date'],
                                   right_on=['ticker', 'simfinid', 'comp_name', 'date'])

            # Select columns
            results_df4 = results_df3[
                ['simfinid', 'comp_name', 'ticker', 'weights', 'month_end_date', 'period_start_date',
                 'period_end_date', 'open_x', 'open_y']].rename(columns={'open_x': 'open_start_date',
                                                                         'open_y': 'open_end_date'})

            # Create initial portfolio amount column
            results_df4.loc[:, 'portfolio_amount'] = initial_amount
            # Create weighted amounts
            results_df4.loc[:, 'begin_dollar_amount'] = (results_df4.loc[:, 'weights'] *
                                                         results_df4.loc[:, 'portfolio_amount'])
            results_df4.loc[:, 'begin_dollar_amount'] = results_df4['begin_dollar_amount'].astype(float)
            # Calculate shares held
            results_df4.loc[:, 'shares_held'] = (results_df4.loc[:, 'begin_dollar_amount'] /
                                                 results_df4.loc[:, 'open_start_date'])
            # Calculate period end dollar amount
            results_df4.loc[:, 'end_dollar_amount'] = (results_df4.loc[:, 'shares_held'] *
                                                       results_df4.loc[:, 'open_end_date'])
            results_df4.loc[:, 'end_dollar_amount'] = results_df4['end_dollar_amount'].astype(float)
            # AGGREGATE RESULTS
            final_results = pd.DataFrame(results_df4.agg({'period_start_date': 'mean',
                                                          'period_end_date': 'mean',
                                                          'begin_dollar_amount': 'sum',
                                                          'end_dollar_amount': 'sum'})
                                         ).T
            # Create return column
            final_results.loc[:, 'period_return'] = (final_results['end_dollar_amount'] /
                                                     final_results['begin_dollar_amount'])

            # Append to dataframe
            results = pd.concat([results, final_results])
            # Carry end_dollar_amount
            initial_amount = float(final_results['end_dollar_amount'].astype(float).values)

        return results
