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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting


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

#%% LIGHTGBM
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm = lgb.LGBMRegressor(boosting_type='dart',
                         n_jobs=23)

# Create the parameter dictionary: params
lgbm_param_grid = {'lgbm_model__n_estimators': np.linspace(50, 250, 30, dtype=int),
                   'lgbm_model__learning_rate': [0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002],
                   'lgbm_model__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'lgbm_model__num_leaves': [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                   'lgbm_model__min_data_in_leaf': np.linspace(2000, 5000, 30, dtype=int)
                   }

# Setup the pipeline steps: steps
lgbm_steps = [("lgbm_model", lgbm)]

# Create the pipeline: xgb_pipeline
lgbm_pipeline = Pipeline(lgbm_steps)

# Perform random search: grid_mae
lgbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_pipeline,
                                                                param_distributions=lgbm_param_grid,
                                                                n_iter=400,
                                                                scoring='neg_root_mean_squared_error',
                                                                cv=tscv,
                                                                verbose=10,
                                                                refit=True
                                                                ),
                                             transformer=StandardScaler()
                                             )
# Fit the estimator
lgbm_randomized.fit(X_train, y_train)  # categorical_feature is auto

#%% Compute metrics
# lgbm_randomized = lgbm_coarse_cv_model
# Print the best parameters and lowest MAE
print("Best estimators found: ", lgbm_randomized.regressor_.best_estimator_)
print("Best parameters found: ", lgbm_randomized.regressor_.best_params_)
print("Lowest RMSE found: ", np.abs(lgbm_randomized.regressor_.best_score_))
# Get feature importance
lgb.plot_importance(lgbm_randomized.regressor_.best_estimator_.named_steps['lgbm_model'], max_num_features=40)
plt.rcParams['figure.figsize'] = (20, 15)
plt.show()
# Look at cv_results
lgbm_cv_results = pd.DataFrame(lgbm_randomized.regressor_.cv_results_)

# Visualize hyperparameters


def visualize_hyperparameter(name):
    plt.scatter(lgbm_cv_results[name], lgbm_cv_results['mean_test_score'], c=['blue'])
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


#%% Load modeling data
modeling_data = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/04_feature/modeling_data.parquet'
)


#%% View output results
holdout_results_data = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/'
    'holdout_results_data.parquet'
)

performance_summary = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/'
    'performance_summary_data.parquet'
)

top_predictions_data = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/'
    'top_predictions_view_data.parquet'
)


#%% PORTFOLIO OPTIMIZATION ##
# Get pricing data
price_data = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/02_intermediate/intro_cleaned_data.parquet'
)

#%% Get pricing data for necessary tickers
prices = price_data[price_data['ticker'].isin(list(top_predictions_data['ticker']))]
prices = prices[['date', 'ticker', 'market_cap', 'adj_close']]

# Make time series
prices_2 = prices.pivot(index='date', columns='ticker', values='adj_close')
# Make risk model
S = risk_models.CovarianceShrinkage(prices_2).ledoit_wolf()

#%% Get views and confidences
predictions = top_predictions_data.copy()
predictions = predictions.set_index('date')
predictions.loc[:, 'views'] = predictions['predictions'] - 1
predictions.loc[:, 'confidences'] = 0.9
predictions = predictions[predictions.index.isin(['2020-02-28'])]
# Make dict for views
viewdict = predictions[['ticker', 'views']].reset_index(drop=True).set_index('ticker').to_dict()

#%% Black-litterman
bl = BlackLittermanModel(S,
                         absolute_views=viewdict['views'],
                         omega="idzorek",
                         view_confidences=predictions['confidences']
                         )

#%% Posterior estimate of returns
ret_bl = bl.bl_returns()
print(ret_bl)