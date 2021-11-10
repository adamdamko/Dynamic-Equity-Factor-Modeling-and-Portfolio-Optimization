# This is the production nodes file.

# Import libraries
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import pandas_market_calendars as mcal
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions


class Production:
    """
    The Production class trains and trains and predicts the latest out of sample data.

    """

    @staticmethod
    def production_model(modeling_data: pd.DataFrame, production_train_date: Dict,
                         production_test_date: Dict, lgbm_fine_best_params: Dict,
                         model_features: Dict, model_target: Dict) -> pd.DataFrame:
        """
        This function creates the latest out of sample predictions to be used in
        investing.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            production_train_date: Monthly dates to loop through for validation (likely just one date).
                                   Train on everything less than each date and predict on next date's data.
                                   Use 'LightGBM best fine-tuning parameters' parameter.
                                   This is to be used for real-world investments.
            production_test_date: Test date(s) for production.
            lgbm_fine_best_params: Best-tuned parameters from LightGBM hyperparameter tuning.
            model_features: Final feature set for modeling.
            model_target: Final target variable for modeling.

        Returns:
             Pandas dataframe
        """
        # MODEL

        # CREATE X_TRAIN AND Y_TRAIN
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < production_train_date]
        data_2 = data_2.reset_index(drop=True)
        # Set categorical variables as 'category'
        data_2.loc[:, 'sector'] = data_2.loc[:, 'sector'].astype('category')
        data_2.loc[:, 'market_cap_cat'] = data_2.loc[:, 'market_cap_cat'].astype('category')
        # Extract columns for StandardScaler
        x_train = data_2.copy()
        float_mask = (x_train.dtypes == 'float64')
        # Get list of float column names
        float_columns = x_train.columns[float_mask].tolist()
        # Create StandardScaler object
        scalerx = StandardScaler()
        # Scale columns
        x_train[float_columns] = scalerx.fit_transform(x_train[float_columns])
        # Set variables for data sets
        y_train = data_2[model_target]
        x_train = x_train[model_features]

        # CREATE X_TEST
        data_3 = modeling_data.reset_index(drop=True)
        data_3 = data_3[(data_3['date'] >= production_train_date) & (data_3['date'] < production_test_date)]
        data_3 = data_3.reset_index(drop=True)
        data_3 = data_3.set_index(['date', 'ticker'])
        # Set categorical variables as 'category'
        data_3.loc[:, 'sector'] = data_3.loc[:, 'sector'].astype('category')
        data_3.loc[:, 'market_cap_cat'] = data_3.loc[:, 'market_cap_cat'].astype('category')
        # Extract columns for StandardScaler
        x_test = data_3.copy()
        # Drop target column as it is blank
        x_test = x_test.drop(columns='target_1m_mom_lead')
        float_mask = (x_test.dtypes == 'float64')
        # Get list of float column names
        float_columns = x_test.columns[float_mask].tolist()
        # Scale columns
        x_test[float_columns] = scalerx.fit_transform(x_test[float_columns])
        # Set variables for data sets
        y_test = pd.DataFrame(data_3[model_target])
        x_test = x_test[model_features]

        # LIGHTGBM MODEL
        lgb_model = TransformedTargetRegressor(lgb.LGBMRegressor(
            n_estimators=lgbm_fine_best_params['n_estimators'],
            learning_rate=lgbm_fine_best_params['learning_rate'],
            max_depth=lgbm_fine_best_params['max_depth'],
            num_leaves=lgbm_fine_best_params['num_leaves'],
            min_data_in_leaf=lgbm_fine_best_params['min_data_in_leaf'],
            boosting_type=lgbm_fine_best_params['boosting_type'],
            n_jobs=lgbm_fine_best_params['n_jobs'],
            random_state=lgbm_fine_best_params['random_state']
        ),
            transformer=StandardScaler()
        )

        # Fit the estimator
        lgb_model.fit(x_train, y_train)

        # Test preds
        model_preds = pd.DataFrame(lgb_model.predict(x_test))

        # Join model_preds and y_test
        test_view = pd.concat([y_test, model_preds.set_index(y_test.index)], axis=1)

        # Rename preds column
        final_results = test_view.rename(columns={0: 'predictions'})

        # Get top 100 predictions per month only view
        top_preds_returns = final_results.sort_values(['date', 'predictions'],
                                                      ascending=False).reset_index(level=['date', 'ticker']) \
            .drop(columns='target_1m_mom_lead')
        top_preds = top_preds_returns.groupby('date')['predictions'].nlargest(100).reset_index().set_index('level_1')
        # Join to have dates and tickers included in output
        top_preds_returns = pd.merge(top_preds, top_preds_returns, how='left', left_index=True, right_index=True)
        top_preds_returns = top_preds_returns.drop(columns=['date_x', 'predictions_x']). \
            rename(columns={'date_y': 'date', 'predictions_y': 'predictions'})

        return top_preds_returns

    @staticmethod
    def market_production_dates(production_start_date: Dict, production_end_date: Dict,
                                last_production_date: Dict) -> pd.DataFrame:
        """
        This function creates the schedule of market dates to be used in portfolio optimization.

        Args:
            production_start_date: The first date of the production market schedule.
                                 This date should be the first day of the month before production prediction.
                                 It should be a beginning-of-calendar-month value.
                                 (E.g., if the last prediction date is 2021-10-29 then the production_start_date
                                  would be 2021-10-01).
            production_end_date: The last date of the production market schedule.
                                 This date should be three months beyond the last prediction date.
                                 It should be an end-of-calendar-month value.
                                 (E.g., if the last prediction date is 2021-10-29 then the production_end_date
                                  would be 2022-01-31).
            last_production_date: Date of last prediction. (E.g., if you are predicting for the month of 2021-10
                                  ending 2021-10-29, the last_production_date would be the last trading day of
                                   the prior month, 2021-09-30 in this case).

        Returns:
             Pandas dataframe with a schedule of dates to be used in portfolio optimization.

        """
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        # Get date schedule
        schedule = nyse.schedule(start_date=production_start_date, end_date=production_end_date)
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
        schedule_join = schedule_join[schedule_join['month_end_date'] <= last_production_date]

        return schedule_join

    @staticmethod
    def portfolio_optimization(production_predictions_data: pd.DataFrame, intro_cleaned_data: pd.DataFrame,
                               production_dates_data: pd.DataFrame, confidence_level: Dict) -> pd.DataFrame:
        """
        This function performs portfolio optimization for production.

        Args:
            production_predictions_data: Dataframe with top predictions from model eval.
            intro_cleaned_data: Initial cleaned data from data processing.
            production_dates_data: Schedule of dates to be used in portfolio optimization.
            confidence_level: Confidence level in Black-Litterman optimization (note:
                              this could be different for all stock predictions, but
                              as this is algorithmic, I do not have any reason to have more/less
                               confidence in any given prediction).

        Returns:
             Pandas dataframe with the results from portfolio optimization and backtesting.
        """
        # Set initial parameters
        results = pd.DataFrame()
        confidence = confidence_level

        # PORTFOLIO OPTIMIZATION AND BACKTESTING LOOP
        for date in list(production_dates_data['month_end_date']):
            # Get predictions by month
            preds_data = production_predictions_data[(production_predictions_data['date'] == date)]
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
            results_df2 = pd.merge(results_df, production_dates_data,
                                   how='left',
                                   left_on='date',
                                   right_on='month_end_date',
                                   validate='m:1')

            # Merge with prices for month ahead dates
            results_df3 = pd.merge(results_df2, intro_cleaned_data, how='left',
                                   left_on=['ticker', 'period_start_date'],
                                   right_on=['ticker', 'date'])
            results_df3 = pd.merge(results_df3,
                                   intro_cleaned_data,
                                   how='left',
                                   left_on=['ticker', 'simfinid', 'comp_name', 'period_end_date'],
                                   right_on=['ticker', 'simfinid', 'comp_name', 'date'])

            # Select columns
            results_df4 = results_df3[
                ['simfinid', 'comp_name', 'ticker', 'weights', 'month_end_date', 'period_start_date',
                 'period_end_date', 'open_x']].rename(columns={'open_x': 'open_start_date'})

            # Append to dataframe
            results = pd.concat([results, results_df4])

        return results
