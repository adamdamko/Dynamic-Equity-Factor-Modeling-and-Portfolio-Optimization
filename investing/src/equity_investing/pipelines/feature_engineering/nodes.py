# This is the feature_engineering nodes file.
# Import libraries
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from pandas_profiling import ProfileReport


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
    def filter_dates(filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function removes missing data from January 2007 and February 2020 as these
        are not full months and full months are needed for this analysis.

        Args:
            filtered_data:

        Returns:
            Pandas dataframe of data  '2007-02-01' <= data <= '2021-08-31'
        """
        data = filtered_data
        data_2 = data[(data['date'] >= '2007-02-01') & (data['date'] <= '2021-08-31')]

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
        data.loc[:, 'market_cap'] = data['adj_close'] * data['shares_out']
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
    def create_returns(filtered_dates_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates a return series.

        Args:
            filtered_dates_data:

        Returns:
            Pandas dataframe with a return series
            by company (ticker).
        """
        data = filtered_dates_data
        # Create returns series
        data = data.sort_values(['ticker', 'date'])
        data['return_index'] = data['adj_close'] / data.groupby('ticker')['adj_close'].transform('first')

        return data

    @staticmethod
    def create_rolling_values(returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates rolling values for returns and volume.

        Args:
            returns_data:

        Returns:
            Pandas dataframe with rolling values.
        """
        data = returns_data

        # Create rolling returns for 1-month momentum
        data_2 = data
        data_2.loc[:, 'roll_7day_sum_ret'] = data_2.groupby('ticker')['return_index']. \
            transform(lambda group: group.rolling(7, 7).sum())

        # Create rolling returns for 3-, 6-, 12-month momentum
        data_2 = data
        data_2.loc[:, 'roll_11day_sum_ret'] = data_2.groupby('ticker')['return_index']. \
            transform(lambda group: group.rolling(11, 11).sum())

        # Create 30-day rolling volume
        data_2.loc[:, 'roll_30day_med_vol'] = data_2.groupby('ticker')['volume']. \
            transform(lambda group: group.rolling(30, 10).median())

        return data_2

    @staticmethod
    def market_schedule() -> pd.DataFrame:
        """
        This function creates a NYSE market schedule that will be
        used for joining.

        Args:
            -

        Returns:
            Pandas dataframe with a the NYSE trading day schedule from
            2007-02-01 to 2020-01-31.
        """
        # Get market calendar dates
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        # Get date schedule
        schedule = nyse.schedule(start_date='2007-02-01', end_date='2021-08-31')
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
    def lagged_rolling_returns(rolling_values_data: pd.DataFrame, market_schedule_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the lag rolling return value correctly for
        month over month values. It uses the exact number of trading days
        in a month to calculate the monthly lag.

        Args:
            rolling_values_data:
            market_schedule_data:

        Returns:
            Pandas dataframe with with complete months of rolling returns,
            rolling volume median, and lagged rolling returns.
        """
        # Rename inputs
        data_2 = rolling_values_data
        schedule_join = market_schedule_data

        # Split date into year, month, day
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

        # Create lagged rolling return value
        conditions_1m = [
            (data_4['days_in_month'] == 19),
            (data_4['days_in_month'] == 20),
            (data_4['days_in_month'] == 21),
            (data_4['days_in_month'] == 22),
            (data_4['days_in_month'] == 23)
        ]

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

        choices_1m = [
            (data_4.groupby('ticker')['roll_7day_sum_ret'].shift(16)),
            (data_4.groupby('ticker')['roll_7day_sum_ret'].shift(17)),
            (data_4.groupby('ticker')['roll_7day_sum_ret'].shift(18)),
            (data_4.groupby('ticker')['roll_7day_sum_ret'].shift(19)),
            (data_4.groupby('ticker')['roll_7day_sum_ret'].shift(20))
        ]

        choices_3m = [
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(55)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(55)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(55)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(56)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(57)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(56)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(56)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(56)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(57)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(58)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(57)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(57)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(57)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(58)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(59)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(58)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(58)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(58)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(59)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(60)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(59)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(59)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(59)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(60)),
            (data_4.groupby('ticker')['roll_11day_sum_ret'].shift(61))
        ]
        # Add the lag rolling sum for 1-month momentum
        data_4.loc[:, 'roll_7day_sum_ret_1m_lag'] = np.select(conditions_1m, choices_1m, default=np.nan)
        # Add the lag rolling sum for 3-month momentum
        data_4.loc[:, 'roll_11day_sum_ret_3m_lag'] = np.select(conditions_3m, choices_3m, default=np.nan)
        # Add the lag rolling sum for 6-month momentum
        data_4.loc[:, 'roll_11day_sum_ret_6m_lag'] = data_4.groupby('ticker')['roll_11day_sum_ret'].shift(121)
        # Add the lag rolling sum for 6-month momentum
        data_4.loc[:, 'roll_11day_sum_ret_12m_lag'] = data_4.groupby('ticker')['roll_11day_sum_ret'].shift(247)

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
        data_4 = data_3.loc[data_3['roll_30day_med_vol'] > 20000]

        return data_4

    @staticmethod
    def create_momentum_factors(monthly_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the momentum and lagged momentum factors.

        Args:
            monthly_data:

        Returns:
            Pandas dataframe with momentum factors, complete months of rolling returns,
            rolling volume median, and lagged rolling returns.
        """
        data_3 = monthly_data

        # Set data
        data_4 = data_3[['date',
                         'ticker',
                         'sector',
                         'market_cap',
                         'market_cap_cat',
                         'roll_30day_med_vol',
                         'return_index',
                         'roll_7day_sum_ret',
                         'roll_11day_sum_ret',
                         'roll_7day_sum_ret_1m_lag',
                         'roll_11day_sum_ret_3m_lag',
                         'roll_11day_sum_ret_6m_lag',
                         'roll_11day_sum_ret_12m_lag']]

        # Create 1-month momentum
        data_4.loc[:, 'mom_1_0'] = data_4['roll_7day_sum_ret'] / data_4['roll_7day_sum_ret_1m_lag']
        # Drop outlier 1-month momentum values
        # data_4 = data_4[~(data_4['mom_1_0'].ge(2))]

        # Create 3-month momentum
        data_4.loc[:, 'mom_3_0'] = data_4['roll_11day_sum_ret'] / data_4['roll_11day_sum_ret_3m_lag']
        # Drop outlier 3-month momentum values
        # data_4 = data_4[~(data_4['mom_3_0'].ge(2))]

        # Create 6-month momentum
        data_4.loc[:, 'mom_6_0'] = data_4['roll_11day_sum_ret'] / data_4['roll_11day_sum_ret_6m_lag']
        # Drop outlier 6-month momentum values
        # data_4 = data_4[~(data_4['mom_6_0'].ge(2))]

        # Create 12-month momentum
        data_4.loc[:, 'mom_12_0'] = data_4['roll_11day_sum_ret'] / data_4['roll_11day_sum_ret_12m_lag']
        # Drop outlier 12-month momentum values
        # data_4 = data_4[~(data_4['mom_12_0'].ge(2))]

        # Create 1-month momentum lags
        data_4.loc[:, 'mom_1_0_L1'] = data_4.groupby('ticker')['mom_1_0'].shift(1)
        data_4.loc[:, 'mom_1_0_L2'] = data_4.groupby('ticker')['mom_1_0'].shift(2)
        data_4.loc[:, 'mom_1_0_L3'] = data_4.groupby('ticker')['mom_1_0'].shift(3)
        data_4.loc[:, 'mom_1_0_L4'] = data_4.groupby('ticker')['mom_1_0'].shift(4)
        data_4.loc[:, 'mom_1_0_L5'] = data_4.groupby('ticker')['mom_1_0'].shift(5)
        data_4.loc[:, 'mom_1_0_L6'] = data_4.groupby('ticker')['mom_1_0'].shift(6)
        data_4.loc[:, 'mom_1_0_L7'] = data_4.groupby('ticker')['mom_1_0'].shift(7)
        data_4.loc[:, 'mom_1_0_L8'] = data_4.groupby('ticker')['mom_1_0'].shift(8)
        data_4.loc[:, 'mom_1_0_L9'] = data_4.groupby('ticker')['mom_1_0'].shift(9)
        data_4.loc[:, 'mom_1_0_L10'] = data_4.groupby('ticker')['mom_1_0'].shift(10)
        data_4.loc[:, 'mom_1_0_L11'] = data_4.groupby('ticker')['mom_1_0'].shift(11)
        data_4.loc[:, 'mom_1_0_L12'] = data_4.groupby('ticker')['mom_1_0'].shift(12)
        data_4.loc[:, 'mom_1_0_L13'] = data_4.groupby('ticker')['mom_1_0'].shift(13)
        data_4.loc[:, 'mom_1_0_L14'] = data_4.groupby('ticker')['mom_1_0'].shift(14)
        data_4.loc[:, 'mom_1_0_L15'] = data_4.groupby('ticker')['mom_1_0'].shift(15)
        data_4.loc[:, 'mom_1_0_L16'] = data_4.groupby('ticker')['mom_1_0'].shift(16)
        data_4.loc[:, 'mom_1_0_L17'] = data_4.groupby('ticker')['mom_1_0'].shift(17)
        data_4.loc[:, 'mom_1_0_L18'] = data_4.groupby('ticker')['mom_1_0'].shift(18)
        data_4.loc[:, 'mom_1_0_L19'] = data_4.groupby('ticker')['mom_1_0'].shift(19)
        data_4.loc[:, 'mom_1_0_L20'] = data_4.groupby('ticker')['mom_1_0'].shift(20)
        data_4.loc[:, 'mom_1_0_L21'] = data_4.groupby('ticker')['mom_1_0'].shift(21)
        data_4.loc[:, 'mom_1_0_L22'] = data_4.groupby('ticker')['mom_1_0'].shift(22)
        data_4.loc[:, 'mom_1_0_L23'] = data_4.groupby('ticker')['mom_1_0'].shift(23)
        data_4.loc[:, 'mom_1_0_L24'] = data_4.groupby('ticker')['mom_1_0'].shift(24)

        return data_4

    @staticmethod
    def create_modeling_data(momentum_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates a modeling data set with a categorical
        market cap variable. It creates the target variable as the
        one-month leading monthly momentum variable (next months
        return). Deletes first row of each ticker because
        there are no lagged values of returns (return index). Deletes
        last row as it has no target.

        Args:
            momentum_data:

        Returns:
            Pandas dataframe with target variable ready for modeling.
        """
        data = momentum_data

        # Drop unnecessary columns
        data_2 = data.drop(columns=['market_cap',
                                    'return_index',
                                    'roll_7day_sum_ret',
                                    'roll_11day_sum_ret',
                                    'roll_7day_sum_ret_1m_lag',
                                    'roll_11day_sum_ret_3m_lag',
                                    'roll_11day_sum_ret_6m_lag',
                                    'roll_11day_sum_ret_12m_lag',
                                    ])

        # Create target, move back one-month momentum (which is one-month return)
        data_2.loc[:, 'target_1m_mom_lead'] = data_2.groupby('ticker')['mom_1_0'].shift(-1)

        # Drop first row as it contains no lagged values
        data_3 = data_2.groupby('ticker').apply(lambda group: group.iloc[1:]).reset_index(drop=True)
        # Drop last row as it has no target
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
