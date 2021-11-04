# This is the feature_engineering nodes file.
# Import libraries
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pandas_profiling import ProfileReport
from typing import Dict


class DataFiltering:
    """
    This class performs filtering.

    """

    @staticmethod
    def filter_data(market_cap_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function removes missing data and companies without market cap values.
        Deeper work could be done to assess reasons for missing data. But for this
        modeling, all variables are needed.

        Args:
            market_cap_data:

        Returns:
            Pandas dataframe of cleaned data. There should be no missing values in this
            returned data set. Also, companies with missing market cap values and
            ultra small companies ('penny') stocks are removed.
        """
        data = market_cap_data
        cap_list = list(['other', 'penny'])
        data_2 = data.loc[~data['market_cap_cat'].isin(cap_list)]
        data_2 = data_2.dropna()

        return data_2

    @staticmethod
    def filter_dates(filtered_data: pd.DataFrame, current_date: Dict) -> pd.DataFrame:
        """
        This function removes missing data from January 2007 and February 2020 as these
        are not full months and full months are needed for this analysis.

        Args:
            filtered_data:
            current_date: Last end-of-month date available in data set
             (e.g. if it is 11/1/21 then current_date is 10/29/21).

        Returns:
            Pandas dataframe of data  '2007-02-01' <= data <= '2021-08-31'
        """
        data = filtered_data
        data_2 = data[(data['date'] >= '2007-02-01') & (data['date'] <= current_date)]

        return data_2


class FeatureEngineering:
    """
    This class holds all feature engineering methods.

    """
    @staticmethod
    def market_cap(intro_cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates market cap categories.

        Args:
            intro_cleaned_data:

        Returns:
            Pandas dataframe of data with engineered market
            cap categories.
        """
        data = intro_cleaned_data
        # Make market cap feature
        # data.loc[:, 'market_cap'] = data['adj_close'] * data['shares_out']
        # Make market cap category
        col = 'market_cap'
        conditions = [data[col] >= 2 * 10 ** 11,
                      (data[col] < 2 * 10 ** 11) & (data[col] >= 1 * 10 ** 10),
                      (data[col] < 1 * 10 ** 10) & (data[col] >= 2 * 10 ** 9),
                      (data[col] < 2 * 10 ** 9) & (data[col] >= 3 * 10 ** 8),
                      (data[col] < 3 * 10 ** 8) & (data[col] >= 5 * 10 ** 7),
                      data[col] < 5 * 10 ** 7]
        choices = ['mega', 'large', 'mid', 'small', 'micro', 'penny']
        data.loc[:, 'market_cap_cat'] = np.select(conditions, choices, default='other')

        return data

    @staticmethod
    def create_returns(filtered_dates_data: pd.DataFrame, change_index_list: Dict) -> pd.DataFrame:
        """
        This function creates a return series.

        Args:
            filtered_dates_data:
            change_index_list: list of variables excluding prices for which to make a return index

        Returns:
            Pandas dataframe with a return series
            by company (ticker).
        """
        data = filtered_dates_data
        # Create returns series
        data = data.sort_values(['ticker', 'date'])
        data.loc[:, 'return_index'] = data['adj_close'] / data.groupby('ticker')['adj_close'].transform('first')
        # Create change index
        for feature in change_index_list:
            data.loc[:, feature+'_index'] = data[feature] / data.groupby('ticker')[feature].transform('first')

        return data

    @staticmethod
    def create_rolling_values(returns_data: pd.DataFrame, sum_list: Dict, avg_list: Dict, med30_list: Dict)\
            -> pd.DataFrame:
        """
        This function creates rolling values for returns and volume.

        Args:
            returns_data:
            sum_list: list of columns to make rolling sums
            avg_list: list of columns to make rolling averages
            med30_list: list of columns to make 30-day rolling medians

        Returns:
            Pandas dataframe with rolling values.
        """
        data_2 = returns_data
        ## ROLLING 7-DAY FEATURES
        # Create rolling 7-day sums
        for feature in sum_list:
            data_2.loc[:, 'roll_7day_sum_'+feature] = data_2.groupby('ticker')[feature].\
                transform(lambda group: group.rolling(7, 7).sum())
        # Create rolling 7-day averages
        for feature in avg_list:
            data_2.loc[:, 'roll_7day_avg_'+feature] = data_2.groupby('ticker')[feature].\
                transform(lambda group: group.rolling(7, 7).mean())
        ## ROLLING 11-DAY FEATURES
        # Create rolling 11-day sums
        for feature in sum_list:
            data_2.loc[:, 'roll_11day_sum_'+feature] = data_2.groupby('ticker')[feature].\
                transform(lambda group: group.rolling(11, 11).sum())
        # Create rolling 11-day averages
        for feature in avg_list:
            data_2.loc[:, 'roll_11day_avg_'+feature] = data_2.groupby('ticker')[feature].\
                transform(lambda group: group.rolling(11, 11).mean())
        ## ROLLING 30-DAY FEATURES
        # Create 30-day rolling features
        for feature in med30_list:
            data_2.loc[:, 'roll_30day_med_'+feature] = data_2.groupby('ticker')[feature].\
                transform(lambda group: group.rolling(30, 11).median())

        return data_2

    @staticmethod
    def market_schedule(current_date: Dict) -> pd.DataFrame:
        """
        This function creates a NYSE market schedule that will be
        used for joining.

        Args:
            current_date: Last end-of-month date available in data set
             (e.g. if it is 11/1/21 then current_date is 10/29/21).

        Returns:
            Pandas dataframe with a the NYSE trading day schedule from
            2007-02-01 to 2021-09-30.
        """
        # Get market calendar dates
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        # Get date schedule
        schedule = nyse.schedule(start_date='2007-02-01', end_date=current_date)
        # Get month end
        schedule['year'] = schedule['market_close'].dt.year
        schedule['month'] = schedule['market_close'].dt.month
        schedule['day'] = schedule['market_close'].dt.day
        # Get number of days in group
        schedule['cal_days_in_month'] = schedule.groupby(['year', 'month'])['day'].transform('count')
        # Grab month end dates
        schedule_end = schedule.groupby(['year', 'month'])['day'].max().reset_index()
        # Make schedule to join
        schedule_join = schedule[['year', 'month', 'cal_days_in_month']]
        schedule_join = schedule_join.drop_duplicates().reset_index().drop('index', axis=1)

        return schedule_join

    @staticmethod
    def lagged_rolling_returns(rolling_values_data: pd.DataFrame, market_schedule_data: pd.DataFrame,
                               sum_list: Dict, avg_list: Dict) -> pd.DataFrame:
        """
        This function creates the lag rolling return value correctly for
        month over month values. It uses the exact number of trading days
        in a month to calculate the monthly lag.

        Args:
            rolling_values_data:
            market_schedule_data:
            sum_list: list of columns to make rolling sums
            avg_list: list of columns to make rolling averages

        Returns:
            Pandas dataframe with with complete months of rolling returns,
            rolling volume median, and lagged rolling returns.
        """
        # Rename inputs
        data_2 = rolling_values_data
        schedule_join = market_schedule_data

        # Split date into year, month, day
        data_2.loc[:, 'date'] = pd.to_datetime(data_2['date'])
        data_2.loc[:, 'year'] = data_2['date'].dt.year
        data_2.loc[:, 'month'] = data_2['date'].dt.month
        data_2.loc[:, 'day'] = data_2['date'].dt.day
        # Get number of days in group
        data_2.loc[:, 'days_in_month'] = data_2.groupby(['ticker',
                                                         'year',
                                                         'month'])['day'].transform('count')

        # Join rolling returns to market schedule
        data_3 = pd.merge(data_2, schedule_join, how='left', on=['year', 'month'], validate='m:1')
        # Filter out incomplete months
        data_4 = data_3[data_3['days_in_month'] == data_3['cal_days_in_month']]

        ## CREATE 1-MONTH FEATURES
        # Conditions for 1-month
        conditions_1m = [
            (data_4['days_in_month'] == 19),
            (data_4['days_in_month'] == 20),
            (data_4['days_in_month'] == 21),
            (data_4['days_in_month'] == 22),
            (data_4['days_in_month'] == 23)
        ]
        # Loop through sum_list
        for feature in sum_list:
            choices_1m = [
                (data_4.groupby('ticker')['roll_7day_sum_'+feature].shift(16)),
                (data_4.groupby('ticker')['roll_7day_sum_'+feature].shift(17)),
                (data_4.groupby('ticker')['roll_7day_sum_'+feature].shift(18)),
                (data_4.groupby('ticker')['roll_7day_sum_'+feature].shift(19)),
                (data_4.groupby('ticker')['roll_7day_sum_'+feature].shift(20))
            ]
            # Add the lag rolling sum 1-month feature
            data_4.loc[:, 'roll_7day_sum_'+feature+'_1m_lag'] = np.select(conditions_1m, choices_1m, default=np.nan)

        # Loop through avg_list
        for feature in avg_list:
            choices_1m = [
                (data_4.groupby('ticker')['roll_7day_avg_'+feature].shift(16)),
                (data_4.groupby('ticker')['roll_7day_avg_'+feature].shift(17)),
                (data_4.groupby('ticker')['roll_7day_avg_'+feature].shift(18)),
                (data_4.groupby('ticker')['roll_7day_avg_'+feature].shift(19)),
                (data_4.groupby('ticker')['roll_7day_avg_'+feature].shift(20))
            ]
            # Add the lag rolling average for 1-month feature
            data_4.loc[:, 'roll_7day_avg_'+feature+'_1m_lag'] = np.select(conditions_1m, choices_1m, default=np.nan)

        ## CREATE 3-MONTH FEATURES
        # Conditions for 3-month
        conditions_3m = [
            (data_4['days_in_month'] == 19) & (data_4['days_in_month'].shift(47) == 19),
            (data_4['days_in_month'] == 19) & (data_4['days_in_month'].shift(47) == 20),
            (data_4['days_in_month'] == 19) & (data_4['days_in_month'].shift(47) == 21),
            (data_4['days_in_month'] == 19) & (data_4['days_in_month'].shift(47) == 22),
            (data_4['days_in_month'] == 19) & (data_4['days_in_month'].shift(47) == 23),
            (data_4['days_in_month'] == 20) & (data_4['days_in_month'].shift(47) == 19),
            (data_4['days_in_month'] == 20) & (data_4['days_in_month'].shift(47) == 20),
            (data_4['days_in_month'] == 20) & (data_4['days_in_month'].shift(47) == 21),
            (data_4['days_in_month'] == 20) & (data_4['days_in_month'].shift(47) == 22),
            (data_4['days_in_month'] == 20) & (data_4['days_in_month'].shift(47) == 23),
            (data_4['days_in_month'] == 21) & (data_4['days_in_month'].shift(47) == 19),
            (data_4['days_in_month'] == 21) & (data_4['days_in_month'].shift(47) == 20),
            (data_4['days_in_month'] == 21) & (data_4['days_in_month'].shift(47) == 21),
            (data_4['days_in_month'] == 21) & (data_4['days_in_month'].shift(47) == 22),
            (data_4['days_in_month'] == 21) & (data_4['days_in_month'].shift(47) == 23),
            (data_4['days_in_month'] == 22) & (data_4['days_in_month'].shift(47) == 19),
            (data_4['days_in_month'] == 22) & (data_4['days_in_month'].shift(47) == 20),
            (data_4['days_in_month'] == 22) & (data_4['days_in_month'].shift(47) == 21),
            (data_4['days_in_month'] == 22) & (data_4['days_in_month'].shift(47) == 22),
            (data_4['days_in_month'] == 22) & (data_4['days_in_month'].shift(47) == 23),
            (data_4['days_in_month'] == 23) & (data_4['days_in_month'].shift(47) == 19),
            (data_4['days_in_month'] == 23) & (data_4['days_in_month'].shift(47) == 20),
            (data_4['days_in_month'] == 23) & (data_4['days_in_month'].shift(47) == 21),
            (data_4['days_in_month'] == 23) & (data_4['days_in_month'].shift(47) == 22),
            (data_4['days_in_month'] == 23) & (data_4['days_in_month'].shift(47) == 23)
        ]

        # Loop through sum_list
        for feature in sum_list:
            choices_3m = [
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(60)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(60)),
                (data_4.groupby('ticker')['roll_11day_sum_'+feature].shift(61))
            ]
            # Add the lag rolling sum 3-month feature
            data_4.loc[:, 'roll_11day_sum_'+feature+'_3m_lag'] = np.select(conditions_3m, choices_3m, default=np.nan)

        # Loop through avg_list
        for feature in avg_list:
            choices_3m = [
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(55)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(56)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(57)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(58)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(60)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(59)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(60)),
                (data_4.groupby('ticker')['roll_11day_avg_' + feature].shift(61))
            ]
            # Add the lag rolling sum 3-month feature
            data_4.loc[:, 'roll_11day_avg_'+feature+'_3m_lag'] = np.select(conditions_3m, choices_3m, default=np.nan)

        ## ROLLING 6-MONTH FEATURES
        # Create 6-month sum features
        for feature in sum_list:
            data_4.loc[:, 'roll_11day_sum_'+feature+'_6m_lag'] = data_4.groupby('ticker')['roll_11day_sum_'+feature].\
                shift(121)
        # Create 6-month avg features
        for feature in avg_list:
            data_4.loc[:, 'roll_11day_avg_'+feature+'_6m_lag'] = data_4.groupby('ticker')['roll_11day_avg_'+feature]. \
                shift(121)

        ## ROLLING 12-MONTH FEATURES
        # Create 12-month sum features
        for feature in sum_list:
            data_4.loc[:, 'roll_11day_sum_'+feature+'_12m_lag'] = data_4.groupby('ticker')['roll_11day_sum_'+feature].\
                shift(247)
        # Create 12-month avg features
        for feature in avg_list:
            data_4.loc[:, 'roll_11day_avg_'+feature+'_12m_lag'] = data_4.groupby('ticker')['roll_11day_avg_'+feature]. \
                shift(247)

        return data_4

    @staticmethod
    def get_monthly(lagged_rolling_returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function filters to month end dates only.

        Args:
            lagged_rolling_returns_data:

        Returns:
            Pandas dataframe with month end data.
        """
        data = lagged_rolling_returns_data
        # Get last day in each month
        data_2 = data.groupby(['ticker', 'year', 'month'])['day'].max().reset_index()
        # Merge last day with full dataset to get only last day of month
        data_3 = pd.merge(data_2, data, how='left', on=['ticker', 'year', 'month', 'day'], validate='1:1')
        # Remove low volume stocks
        data_4 = data_3.loc[data_3['roll_30day_med_volume'] > 20000]

        return data_4

    @staticmethod
    def create_momentum_factors(monthly_data: pd.DataFrame, param_features: Dict, mom_1m_numerators_list: Dict,
                                mom_numerators_list: Dict, mom_1m_denominators_list: Dict,
                                mom_3m_denominators_list: Dict, mom_6m_denominators_list: Dict,
                                mom_12m_denominators_list: Dict, mom_feature_names_list: Dict,
                                lag_range) -> pd.DataFrame:
        """
        This function creates the momentum and lagged momentum factors.
        Outlier dropping was tried but resulted in weaker performance during times the market was flat
        to down. Old commented code for outlier filtering has been dropped.

        Args:
            monthly_data: Output from get_monthly method.
            param_features: Feature list for the data set.
            mom_1m_numerators_list: List for one-month momentum numerator values (separate because
                                    these will have lagged values).
                                    Each value in this list must coincide with a value in the
                                    'mom_1m_denominators_list'.
            mom_numerators_list: List of 3-, 6-, and 12-month momentum numerator values.
                                 Each value in this list must coincide with a value in the
                                 corresponding 'mom_denominators_list'.
            mom_1m_denominators_list: List for one-month momentum denominator values (separate because
                                      these will have lagged values).
                                      Each value in these lists must coincide with a value in the
                                      'mom_1m_numerators_list'.
            mom_3m_denominators_list: List of 3-month momentum denominator values.
                                      Each value in these lists must coincide with a value in the
                                      'mom_numerators_list'.
            mom_6m_denominators_list: List of 6-month momentum denominator values.
                                      Each value in these lists must coincide with a value in the
                                      'mom_numerators_list'.
            mom_12m_denominators_list: List of 12-month momentum denominator values.
                                       Each value in these lists must coincide with a value in the
                                       'mom_numerators_list'.
            mom_feature_names_list: List of resulting feature names from momentum feature creation.
                                    Each value in this list must coincide with a value in the
                                    'mom_numerators_list' and 'mom_denominators_list'.
            lag_range: Top of range for 1-month lagged variables (i.e., 25 means lags 1 to 24).


        Returns:
            Pandas dataframe with momentum factors, complete months of rolling values,
            rolling volume median, and lagged rolling values.
        """
        data_3 = monthly_data

        # Set data
        data_4 = data_3[param_features]

        ## CREATE MOMENTUM FACTORS
        # Create 1-month momentum.
        for num, denom, name in list(zip(mom_1m_numerators_list, mom_1m_denominators_list, mom_feature_names_list)):
            data_4.loc[:, name + '_mom_1_0'] = data_4[num] / data_4[denom]

        # Create 3-month momentum
        for num, denom, name in list(zip(mom_numerators_list, mom_3m_denominators_list, mom_feature_names_list)):
            data_4.loc[:, name + '_mom_3_0'] = data_4[num] / data_4[denom]

        # Create 6-month momentum
        for num, denom, name in list(zip(mom_numerators_list, mom_6m_denominators_list, mom_feature_names_list)):
            data_4.loc[:, name + '_mom_6_0'] = data_4[num] / data_4[denom]

        # Create 12-month momentum
        for num, denom, name in list(zip(mom_numerators_list, mom_12m_denominators_list, mom_feature_names_list)):
            data_4.loc[:, name + '_mom_12_0'] = data_4[num] / data_4[denom]

        # Create 1-month momentum lags
        for name in mom_feature_names_list:
            for num in range(1, lag_range):
                data_4.loc[:, name + '_mom_1_0_L' + str(num)] = data_4.groupby('ticker')[name + '_mom_1_0'].shift(num)

        return data_4

    @staticmethod
    def create_modeling_data(momentum_data: pd.DataFrame, modeling_data_drop_list: Dict,
                             model_target: Dict) -> pd.DataFrame:
        """
        This function creates a modeling data set with a categorical
        market cap variable. It creates the target variable as the
        one-month leading monthly momentum variable (next months
        return). Deletes first row of each ticker because
        there are no lagged values of returns (return index). Deletes
        last row as it has no target.

        Args:
            momentum_data: Output of 'create_momentum_factors' method
            modeling_data_drop_list: List of variables to drop from final data set.
            model_target: Model target variable.

        Returns:
            Pandas dataframe with target variable ready for modeling.
        """
        data = momentum_data

        # Drop unnecessary columns
        data_2 = data.drop(columns=modeling_data_drop_list)

        # Create target, move back one-month momentum (which is one-month return)
        data_2.loc[:, model_target] = data_2.groupby('ticker')['return_mom_1_0'].shift(-1)

        # Drop first row as it contains no lagged values
        data_3 = data_2.groupby('ticker').apply(lambda group: group.iloc[1:]).reset_index(drop=True)
        # Drop last 3 rows as they have no target
        data_4 = data_3.groupby('ticker').apply(lambda group: group.iloc[:-1]).reset_index(drop=True)
        # Drop missing values
        data_5 = data_4.dropna()

        return data_5

    @staticmethod
    def create_final_eda_data(modeling_data: pd.DataFrame):
        """
        This function creates an html eda file for the modeling data set.

        Args:
            modeling_data:

        Returns:
            Pandas profiling html file for viewing.
        """
        eda_profile = ProfileReport(modeling_data, title='Equity Investing Final Modeling Data EDA', explorative=True)
        # As json file
        eda_profile_html = eda_profile.to_html()

        return eda_profile_html
