# THIS IS THE TESTING GROUNDS FILE #
# Code is tested here before going into pipelines.
# Once code is in pipeline, it may be deleted from this file.

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


#%% Import libraries
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
import pypfopt
import pandas_market_calendars as mcal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions

#%% Get train data and splits
X_train = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_x_data.parquet'
)
y_train = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_y_data.parquet'
)
tscv = pd.read_pickle(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/time_series_split_list.pickle'
)

#%% Make categorical vars as pd.Categorical
# Train X data
X_train.loc[:, 'sector'] = X_train.loc[:, 'sector'].astype('category')
X_train.loc[:, 'market_cap_cat'] = X_train.loc[:, 'market_cap_cat'].astype('category')

#%% LIGHTGBM COARSE-TUNING RANDOMIZED SEARCH CV
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm_coarse = lgb.LGBMRegressor(boosting_type='dart',
                                n_jobs=23)

# Create the parameter dictionary: params
lgbm_coarse_param_grid = {'lgbm_model__n_estimators': np.linspace(50, 1000, 30, dtype=int),
                          'lgbm_model__learning_rate': [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                                                        0.001, 0.002, 0.005, 0.001, 0.01],
                          'lgbm_model__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                          'lgbm_model__num_leaves': [4, 6, 8, 16, 32, 64, 128, 256, 386, 512, 1024, 2048, 4096, 8192],
                          'lgbm_model__min_data_in_leaf': np.linspace(50, 2500, 30, dtype=int)
                          }

# Setup the pipeline steps: steps
lgbm_coarse_steps = [("lgbm_model", lgbm_coarse)]

# Create the pipeline: xgb_pipeline
lgbm_coarse_pipeline = Pipeline(lgbm_coarse_steps)

# Perform random search: grid_mae
lgbm_coarse_search = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_coarse_pipeline,
                                                                   param_distributions=lgbm_coarse_param_grid,
                                                                   n_iter=1000,
                                                                   scoring='neg_root_mean_squared_error',
                                                                   cv=tscv,
                                                                   verbose=10,
                                                                   refit=True
                                                                   ),
                                                transformer=StandardScaler()
                                                )
# Fit the estimator
lgbm_coarse_search.fit(X_train, y_train)  # categorical_feature is auto

#%% LIGHTGBM FINE-TUNING GRID SEARCH CV
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm_fine = lgb.LGBMRegressor(boosting_type='dart',
                              n_jobs=23)

# Create the parameter dictionary: params
lgbm_fine_param_grid = {'lgbm_model__n_estimators': [250, 350, 450, 550, 650, 750],
                        'lgbm_model__learning_rate': [0.0004, 0.0005, 0.0006, 0.0007],
                        'lgbm_model__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                        'lgbm_model__num_leaves': [4, 6, 8, 16, 32, 64, 128, 256],
                        'lgbm_model__min_data_in_leaf': [150, 250, 750, 850, 950]
                        }

# Setup the pipeline steps: steps
lgbm_fine_steps = [("lgbm_model", lgbm_fine)]

# Create the pipeline: xgb_pipeline
lgbm_fine_pipeline = Pipeline(lgbm_fine_steps)

# Perform random search: grid_mae
lgbm_fine_search = TransformedTargetRegressor(GridSearchCV(estimator=lgbm_fine_pipeline,
                                                           param_grid=lgbm_fine_param_grid,
                                                           scoring='neg_root_mean_squared_error',
                                                           cv=tscv,
                                                           verbose=10,
                                                           refit=True
                                                           ),
                                              transformer=StandardScaler()
                                              )
# Fit the estimator
lgbm_fine_search.fit(X_train, y_train)  # categorical_feature is auto

#%% LIGHTGBM COARSE-TUNING HALVING RANDOMIZED SEARCH CV
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm_halve_coarse = lgb.LGBMRegressor(boosting_type='dart',
                                      n_jobs=23)

# Create the parameter dictionary: params
lgbm_halve_coarse_param_grid = {'lgbm_model__n_estimators': [50, 70, 90, 120, 150, 175, 200, 250, 300,
                                                             350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850,
                                                             900, 950, 1000],
                                'lgbm_model__learning_rate': [0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                                                              0.001, 0.002, 0.005, 0.001, 0.01],
                                'lgbm_model__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                          17, 18, 19, 20],
                                'lgbm_model__num_leaves': [4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 386, 424,
                                                           512, 728, 934,  1024, 1276, 1454, 1687, 1892, 2048,
                                                           4096, 8192],
                                'lgbm_model__min_data_in_leaf': [150, 175, 200, 250, 300, 350, 400, 450, 500, 550,
                                                                 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100,
                                                                 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                                                                 2000, 2100, 2200, 2300, 2400, 2500]
                                }

# Setup the pipeline steps: steps
lgbm_halve_coarse_steps = [("lgbm_model", lgbm_halve_coarse)]

# Create the pipeline: xgb_pipeline
lgbm_halve_coarse_pipeline = Pipeline(lgbm_halve_coarse_steps)

# Perform random search: grid_mae
lgbm_halve_coarse_search = TransformedTargetRegressor(HalvingRandomSearchCV(
    estimator=lgbm_halve_coarse_pipeline,
    param_distributions=lgbm_halve_coarse_param_grid,
    n_candidates=1000,
    factor=2,
    min_resources='exhaust',
    scoring='neg_root_mean_squared_error',
    cv=tscv,
    verbose=10,
    refit=True
    ),
    transformer=StandardScaler()
)

# Fit the estimator
lgbm_halve_coarse_search.fit(X_train, y_train)  # categorical_feature is auto


#%% Compute metrics
lgbm_randomized = lgbm_halve_coarse_search  # change this to the estimator of choice
# Print the best parameters and lowest MAE
print("Best estimators found: ", lgbm_randomized.regressor_.best_estimator_)
print("Best parameters found: ", lgbm_randomized.regressor_.best_params_)
print("Lowest RMSE found: ", np.abs(lgbm_randomized.regressor_.best_score_))
# Get feature importance
lgb.plot_importance(lgbm_randomized.regressor_.best_estimator_.named_steps['lgbm_model'], max_num_features=40)
plt.rcParams['figure.figsize'] = (20, 15)
plt.show()
# Look at cv_results
lgbm_cv_results_halve_coarse = pd.DataFrame(lgbm_randomized.regressor_.cv_results_)

#%% Visualize hyperparameters


def visualize_hyperparameter(name):
    plt.scatter(lgbm_cv_results_halve_coarse[name], lgbm_cv_results_halve_coarse['mean_test_score'], c=['blue'])
    plt.gca().set(xlabel='{}'.format(name),
                  ylabel='RMSE',
                  title='RMSE for different {}s'.format(name))
    plt.gca().set_ylim()
    plt.show()


param_list = ['param_lgbm_model__n_estimators',
              'param_lgbm_model__learning_rate',
              'param_lgbm_model__max_depth',
              'param_lgbm_model__num_leaves',
              'param_lgbm_model__min_data_in_leaf']


for param in param_list:
    visualize_hyperparameter(param)


#%% INITIAL RESULTS ##
# Get initial performance summary
initial_performance_summary = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/'
    'Initial_performance_summary_data.parquet'
)
# Get initial backtesting results
initial_backtesting_results = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/'
    'Initial_backtesting_results_data.parquet'
)

#%% LATEST RESULTS ##
# Get performance summary
performance_summary = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/'
    'performance_summary_data.parquet'
)
# Get backtesting results
backtesting_results = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/backtesting_results_data.parquet'
)
# Get backtesting results
results = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/production_results_data.parquet'
)
# Get predictions
predictions = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/production_predictions_data.parquet'
)
#%% LOOK AT DATES ##
# Get production dates
production_dates = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/production_dates_data.parquet'
)
# Get train dates
backtest_dates = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/market_dates_data.parquet'
)
