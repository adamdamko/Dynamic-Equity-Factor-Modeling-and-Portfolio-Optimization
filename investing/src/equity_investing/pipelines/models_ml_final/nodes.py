# This is the models_ml_final nodes file.

# This is the chosen modeling method (Lightgbm) file.
# Comments on hyperparameters from initial 'coarse' tuning:
#   n_estimators (num_iterations) : best performance n_estimators < 300
#   learning_rate: best performance at < 0.001
#             - ** Convergence issues at learning_rate (<0.01) may be a problem, but do
#               not seem to be currently. This may be something to look into down the road.**
#   max_depth: keep this value lower to reduce over-fitting (5-10)
#   num_leaves: Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number
#               of leaves as depth-wise tree. Unconstrained depth can induce over-fitting.
#             - small number of leaves also reduces over-fitting
#             - keep num_leaves < 2^(max_depth
#   min_data_in_leaf: performance seems to be less volatile at levels > 2000 and <= 5000
#   dart: seems to improve performance
#   random_state: after testing, random_state is relatively stable

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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


class TrainTestValidation:
    """
    The TrainTestValidation class splits data into train, test, and
    validation sets.

    """
    @staticmethod
    def holdout_set(modeling_data: pd.DataFrame, modeling_data_date: Dict) -> pd.DataFrame:
        """
        This function creates the holdout data set.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            modeling_data_date: Date to split train/validation sets from the final holdout set.
                                The date given is the date that starts the holdout set (i.e., the
                                train/validation sets will be all dates less than the date given).

        Returns:
            Pandas holdout dataframe.

        """
        data_2 = modeling_data[modeling_data['date'] >= modeling_data_date]
        data_2.loc[:, 'sector'] = data_2.loc[:, 'sector'].astype('category')
        data_2.loc[:, 'market_cap_cat'] = data_2.loc[:, 'market_cap_cat'].astype('category')
        data_2 = data_2.reset_index(drop=True)
        return data_2

    @staticmethod
    def train_val_x(modeling_data: pd.DataFrame, modeling_data_date: Dict) -> pd.DataFrame:
        """
        This functions creates the predictor variable train/test set.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            modeling_data_date: Date to split train/validation sets from the final holdout set.
                                The date given is the date that starts the holdout set (i.e., the
                                train/validation sets will be all dates less than the date given).

        Returns:
            Pandas dataframe.
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < modeling_data_date]
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
    def train_val_y(modeling_data: pd.DataFrame, modeling_data_date: Dict, model_target: Dict) -> pd.DataFrame:
        """
        This function creates the target train/test variable.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            modeling_data_date: Date to split train/validation sets from the final holdout set.
                                The date given is the date that starts the holdout set (i.e., the
                                train/validation sets will be all dates less than the date given).
            model_target: Model target variable.

        Returns:
             Pandas dataframe
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < modeling_data_date]
        data_2 = data_2.reset_index(drop=True)
        # Train y
        y_train = pd.DataFrame(data_2[model_target])
        # Reset index to match with splits
        y_train = y_train.reset_index(drop=True)

        return y_train

    @staticmethod
    def time_series_split(modeling_data: pd.DataFrame, modeling_data_date: Dict) -> list:
        """
        This function creates a list for time-series splits for cross-validation.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            modeling_data_date: Date to split train/validation sets from the final holdout set.
                                The date given is the date that starts the holdout set (i.e., the
                                train/validation sets will be all dates less than the date given).

        Returns:
            List of indices for time-series splits used for cross-validation.
        """
        data_2 = modeling_data.reset_index(drop=True)
        data_2 = data_2[data_2['date'] < modeling_data_date]
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
        test_4 = (train_data[(train_data['date'] >= '2019-07-01') & (train_data['date'] < modeling_data_date)]).index

        # Put train and validation sets in lists
        train_list = [train_1, train_2, train_3, train_4]
        test_list = [test_1, test_2, test_3, test_4]

        # Create generator to be used in modeling
        split_list = [[n, m] for n, m in zip(train_list, test_list)]

        return split_list


class HyperparameterTuning:
    """
    The HyperparameterTuning class cross-validates across the time_series_splits
    to optimize hyperparameters for the LightGBM model.
    """
    @staticmethod
    def train_cv_lgbm(train_x_data: pd.DataFrame, train_y_data: pd.DataFrame, time_series_split_list: list,
                      params_static: Dict, params: Dict, params_features: Dict):
        """
        This performs cross-validation for hyperparameter tuning.

        Args:
            train_x_data:
            train_y_data:
            time_series_split_list:
            params_static: Static parameters defined in parameters.yml
            params: Parameters defined in parameters.yml
            params_features: model train features

        Returns:
            Model object.

        """
        # Make categorical vars as pd.Categorical
        X_train = train_x_data.copy()
        y_train = train_y_data.copy()
        tscv = time_series_split_list.copy()
        # Get features
        X_train = X_train[params_features]
        # Train X data
        X_train.loc[:, 'sector'] = X_train.loc[:, 'sector'].astype('category')
        X_train.loc[:, 'market_cap_cat'] = X_train.loc[:, 'market_cap_cat'].astype('category')

        # LIGHTGBM PIPELINE
        # Instantiate regressor
        lgbm = lgb.LGBMRegressor(boosting_type=params_static['boosting_type'],
                                 extra_trees=params_static['extra_trees'],
                                 n_jobs=params_static['n_jobs']
                                 )

        # Create the parameter dictionary: params
        lgbm_param_grid = params

        # Setup the pipeline steps: steps
        lgbm_steps = [("lgbm_model", lgbm)]

        # Create the pipeline: xgb_pipeline
        lgbm_pipeline = Pipeline(lgbm_steps)

        # Perform random search: grid_mae
        lgbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_pipeline,
                                                                        param_distributions=lgbm_param_grid,
                                                                        n_iter=params_static['n_iter'],
                                                                        scoring=params_static['scoring'],
                                                                        cv=tscv,
                                                                        verbose=10,
                                                                        refit=True
                                                                        ),
                                                     transformer=StandardScaler()
                                                     )
        # Fit the estimator
        lgbm_randomized.fit(X_train, y_train)  # categorical_feature is auto

        return lgbm_randomized

    @staticmethod
    def visualize_hyperparameters(lgbm_cv_model):
        """
        This prints hyperparameter visualization.

        Args:
            lgbm_cv_model: output from lgbm model

        Returns:
            Matplotlib objects.

        """
        # Get results dataframe
        lgbm_cv_results = pd.DataFrame(lgbm_cv_model.regressor_.cv_results_)

        param_list = ['param_lgbm_model__n_estimators',
                      'param_lgbm_model__learning_rate',
                      'param_lgbm_model__max_depth',
                      'param_lgbm_model__num_leaves',
                      'param_lgbm_model__min_data_in_leaf']

        fig, axes = plt.subplots(6)
        for ax, param in zip(axes, param_list):
            ax.scatter(lgbm_cv_results[param], lgbm_cv_results['mean_test_score'], c=['blue'])
            ax.set(xlabel='{}'.format(param),
                   ylabel='RMSE',
                   title='RMSE for different {}s'.format(param))

        return plt

    @staticmethod
    def random_state_test(train_x_data: pd.DataFrame, train_y_data: pd.DataFrame, time_series_split_list: list,
                          params_static: Dict, params: Dict, params_features: Dict):
        """
        This performs random-state cross-validation for tuning.

        Args:
            train_x_data:
            train_y_data:
            time_series_split_list:
            params_static: Static parameters defined in parameters.yml
            params: Parameters defined in parameters.yml
            params_features: Features

        Returns:
            Model object.

        """
        # Make categorical vars as pd.Categorical
        X_train = train_x_data.copy()
        y_train = train_y_data.copy()
        tscv = time_series_split_list.copy()
        # Train X data
        X_train = X_train[params_features]
        X_train.loc[:, 'sector'] = X_train.loc[:, 'sector'].astype('category')
        X_train.loc[:, 'market_cap_cat'] = X_train.loc[:, 'market_cap_cat'].astype('category')

        # Using best hyperparameters from fine-tune
        # LIGHTGBM PIPELINE
        # Instantiate regressor
        lgbm_seed = lgb.LGBMRegressor(n_estimators=params['n_estimators'],
                                      learning_rate=params['learning_rate'],
                                      num_leaves=params['num_leaves'],
                                      max_depth=params['max_depth'],
                                      min_data_in_leaf=params['min_data_in_leaf'],
                                      boosting_type=params_static['boosting_type'],
                                      n_jobs=params_static['n_jobs']
                                      )

        # Create the parameter dictionary: params
        lgbm_seed_param_grid = {
            'lgbm_model__random_state': np.linspace(1000, 999999, 200, dtype=int)
        }

        # Setup the pipeline steps: steps
        lgbm_seed_steps = [("lgbm_model", lgbm_seed)]

        # Create the pipeline: xgb_pipeline
        lgbm_seed_pipeline = Pipeline(lgbm_seed_steps)

        # Perform random search: grid_mae
        lgbm_seed_randomized = TransformedTargetRegressor(
            RandomizedSearchCV(estimator=lgbm_seed_pipeline,
                               param_distributions=lgbm_seed_param_grid,
                               n_iter=200,
                               scoring=params_static['scoring'],
                               cv=tscv,
                               verbose=10,
                               refit=True
                               ),
            transformer=StandardScaler()
            )
        # Fit the estimator
        lgbm_seed_randomized.fit(X_train, y_train)  # categorical_feature is auto

        return lgbm_seed_randomized
