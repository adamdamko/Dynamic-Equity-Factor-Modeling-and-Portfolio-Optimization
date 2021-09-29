# This is the models nodes file.
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrainTestValidation:
    """
    The TrainTestValidation class splits data into train, test, and
    validation sets.

    """
    @staticmethod
    def holdout_set(modeling_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the holdout data set.

        Args:
            modeling_data:

        Returns:
            Pandas holdout dataframe.

        """
        data_2 = modeling_data[modeling_data['date'] >= '2020-02-01']
        data_2.loc[:, 'sector'] = data_2.loc[:, 'sector'].astype('category')
        data_2.loc[:, 'market_cap_cat'] = data_2.loc[:, 'market_cap_cat'].astype('category')
        data_2 = data_2.reset_index(drop=True)
        return data_2

    @staticmethod
    def train_val_x(modeling_data: pd.DataFrame) -> pd.DataFrame:
        """
        This functions creates the predictor variable train/test set.

        Args:
            modeling_data:

        Returns:
            Pandas dataframe.
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < '2020-02-01']
        data_2 = data_2.reset_index(drop=True)
        # Train X data
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
        # Drop date and ticker columns
        x_train = x_train.drop(columns=['date', 'ticker', 'target_1m_mom_lead'])
        # Reset index to match with splits
        x_train = x_train.reset_index(drop=True)

        return x_train

    @staticmethod
    def train_val_y(modeling_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the target train/test variable.

        Args:
            modeling_data:

        Returns:
             Pandas dataframe
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < '2020-02-01']
        data_2 = data_2.reset_index(drop=True)
        # Train y
        y_train = pd.DataFrame(data_2['target_1m_mom_lead'])
        # Reset index to match with splits
        y_train = y_train.reset_index(drop=True)

        return y_train

    @staticmethod
    def time_series_split(modeling_data: pd.DataFrame) -> list:
        """
        This function creates a list for time-series splits for cross-validation.

        Args:
            modeling_data:

        Returns:
            List of indices for time-series splits used for cross-validation.
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < '2020-02-01']
        data_2 = data_2.reset_index(drop=True)
        train_data = data_2

        # Create train and tests sets
        train_1 = (train_data[train_data['date'] < '2013-01-01']).index
        test_1 = (train_data[(train_data['date'] >= '2015-01-01') & (train_data['date'] < '2016-07-01')]).index
        train_2 = (train_data[train_data['date'] < '2014-01-01']).index
        test_2 = (train_data[(train_data['date'] >= '2016-07-01') & (train_data['date'] < '2018-01-01')]).index
        train_3 = (train_data[train_data['date'] < '2015-01-01']).index
        test_3 = (train_data[(train_data['date'] >= '2018-01-01') & (train_data['date'] < '2019-07-01')]).index
        train_4 = (train_data[train_data['date'] < '2016-01-01']).index
        test_4 = (train_data[(train_data['date'] >= '2019-07-01') & (train_data['date'] < '2020-02-01')]).index

        # Put train and validation sets in lists
        train_list = [train_1, train_2, train_3, train_4]
        test_list = [test_1, test_2, test_3, test_4]

        # Create generator to be used in modeling
        split_list = [[n, m] for n, m in zip(train_list, test_list)]

        return split_list
