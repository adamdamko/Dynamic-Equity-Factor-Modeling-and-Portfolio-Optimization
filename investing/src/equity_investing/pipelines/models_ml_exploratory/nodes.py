# This is the models_ml_exploratory node

# This file is the testing ground file for different modeling
# techniques. Three techniques were tried: Random Forest, XGBoost,
# and Lightgbm. In initial testing Lightgbm showed slightly better
# performance in cross-validation and was much faster. Therefore,
# Lightgbm was chosen for the final modeling technique. To play with
# the below models again dummy variables will have to be created
# for the data sets as Lightgbm has it's own handling of categorical
# variables and the data sets were changed to not create dummy
# variables.

# Running this pipeline will take a very long time.
# **First, you need to go and create dummies for sector and market_cap_cat.**

# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import logging
from typing import Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


class ExploratoryModels:
    """
    The ExploratoryModels class cross-validates across the time_series_splits
    to optimize hyperparameters for competing models. This is basic testing
    to decide on a final modeling method.
    """
    @staticmethod
    def train_cv_rf(train_x_data: pd.DataFrame, train_y_data: pd.DataFrame, time_series_split_list: list,
                    params_static: Dict, params: Dict):
        """
        This performs cross-validation for hyperparameter tuning for a random forest model.

        Args:
            train_x_data:
            train_y_data:
            time_series_split_list:
            params_static: Static parameters defined in parameters.yml
            params: Parameters defined in parameters.yml

        Returns:
            Model object.

        """
        # Make categorical vars as pd.get_dummies
        X_train = train_x_data.copy()
        y_train = train_y_data.copy()
        tscv = time_series_split_list.copy()
        # Train X data
        X_train = pd.get_dummies(data=X_train, columns=['sector', 'market_cap_cat'])

        # Instantiate model
        rfr = RandomForestRegressor(n_jobs=params_static['n_jobs'])

        # Create the parameter dictionary: params
        rfr_param_grid = params

        # Setup the pipeline steps: steps
        rfr_steps = [("rfr_model", rfr)]

        # Create the pipeline: rfr_pipeline
        rfr_pipeline = Pipeline(rfr_steps)

        # Build a random search using param_dist and rfr
        rfr_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=rfr_pipeline,
                                                                       param_distributions=rfr_param_grid,
                                                                       n_iter=params_static['n_iter'],
                                                                       scoring='neg_mean_absolute_percentage_error',
                                                                       cv=tscv,
                                                                       verbose=10,
                                                                       refit=True
                                                                       ),
                                                    transformer=StandardScaler()
                                                    )

        # Fit the estimator
        rfr_randomized.fit(X_train, y_train)

        return rfr_randomized

    @staticmethod
    def train_cv_xgb(train_x_data: pd.DataFrame, train_y_data: pd.DataFrame, time_series_split_list: list,
                     params_static: Dict, params: Dict):
        """
        This performs cross-validation for hyperparameter tuning for an xgboost model.

        Args:
            train_x_data:
            train_y_data:
            time_series_split_list:
            params_static: Static parameters defined in parameters.yml
            params: Parameters defined in parameters.yml

        Returns:
            Model object.

        """
        # Make categorical vars as pd.get_dummies
        X_train = train_x_data.copy()
        y_train = train_y_data.copy()
        tscv = time_series_split_list.copy()
        # Train X data
        X_train = pd.get_dummies(data=X_train, columns=['sector', 'market_cap_cat'])

        # XGBOOST PIPELINE
        # Instantiate regressor
        gbm = xgb.XGBRegressor(booster=params_static['booster'],
                               )

        # Create the parameter dictionary: params
        gbm_param_grid = params

        # Setup the pipeline steps: steps
        gbm_steps = [("xgb_model", gbm)]

        # Create the pipeline: xgb_pipeline
        gbm_pipeline = Pipeline(gbm_steps)

        # Perform random search: grid_mae
        gbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=gbm_pipeline,
                                                                       param_distributions=gbm_param_grid,
                                                                       n_iter=params_static['n_iter'],
                                                                       scoring='neg_mean_absolute_percentage_error',
                                                                       cv=tscv,
                                                                       verbose=10,
                                                                       n_jobs=params_static['n_jobs'],
                                                                       refit=True
                                                                       ),
                                                    transformer=StandardScaler()
                                                    )
        # Fit the estimator
        gbm_randomized.fit(X_train, y_train)

        return gbm_randomized
