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


#%% Get market dates
# Get market calendar dates
# Get NYSE calendar
nyse = mcal.get_calendar('NYSE')
# Get date schedule
schedule = nyse.schedule(start_date='2020-02-28', end_date='2021-12-31')
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
schedule_join = schedule_join[schedule_join['month_end_date'] <= '2021-09-30']

#%% Get pricing data for necessary tickers
results = pd.DataFrame()
initial_amount = 10**5
confidence = 0.9

for date in list(schedule_join['month_end_date']):
    # Get predictions by month
    preds_data = top_predictions_data[(top_predictions_data['date'] == date)]
    # Get price data used for risk model
    prices = price_data[(price_data['date'] <= date) & (price_data['ticker'].isin(list(preds_data['ticker'])))]
    prices = prices[['date', 'ticker', 'market_cap', 'open', 'adj_close']]
    # Make time series of predictions
    prices_2 = prices.pivot(index='date', columns='ticker', values='adj_close').dropna()
    # Make risk model
    S = risk_models.CovarianceShrinkage(prices_2).ledoit_wolf()

    # BLACK-LITTERMAN PORTFOLIO OPTIMIZATION AND ALLOCATION
    # Get views and confidences
    predictions = preds_data.set_index('date')
    # Annualized returns for allocation
    predictions.loc[:, 'views'] = (predictions['predictions'])**12 - 1
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
    results_df2 = pd.merge(results_df, schedule_join,
                           how='left',
                           left_on='date',
                           right_on='month_end_date',
                           validate='m:1')

    # Merge with prices for month ahead dates
    results_df3 = pd.merge(results_df2, price_data, how='left', left_on=['ticker', 'period_start_date'],
                           right_on=['ticker', 'date'])
    results_df3 = pd.merge(results_df3,
                           price_data,
                           how='left',
                           left_on=['ticker', 'simfinid', 'comp_name', 'period_end_date'],
                           right_on=['ticker', 'simfinid', 'comp_name', 'date'])

    # Select columns
    results_df4 = results_df3[['simfinid', 'comp_name', 'ticker', 'weights', 'month_end_date', 'period_start_date',
                               'period_end_date', 'open_x', 'open_y']].rename(columns={'open_x': 'open_start_date',
                                                                                       'open_y': 'open_end_date'})

    # Create initial portfolio amount column
    results_df4.loc[:, 'portfolio_amount'] = initial_amount
    # Create weighted amounts
    results_df4.loc[:, 'begin_dollar_amount'] = results_df4.loc[:, 'weights'] * results_df4.loc[:, 'portfolio_amount']
    results_df4.loc[:, 'begin_dollar_amount'] = results_df4['begin_dollar_amount'].astype(float)
    # Calculate shares held
    results_df4.loc[:, 'shares_held'] = (results_df4.loc[:, 'begin_dollar_amount'] /
                                         results_df4.loc[:, 'open_start_date'])
    # Calculate period end dollar amount
    results_df4.loc[:, 'end_dollar_amount'] = results_df4.loc[:, 'shares_held'] * results_df4.loc[:, 'open_end_date']
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


#%% PLOTTING
# Plot Efficient Frontier
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.show()

#%% Get portfolio weights
ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()

#%% Plot allocation
pd.Series(weights).plot.pie(figsize=(10, 10))
plt.show()

#%% Plot weights
plotting.plot_weights(weights=weights)
plt.show()

#%% PORTFOLIO OPTIMIZATION VIEW ##
# Get backtesting results
backtesting_results = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/08_reporting/backtesting_results_data.parquet'
)

#%%
backtesting_results.loc[:, 'rolling_returns'] = backtesting_results['period_return']\
    .rolling(window=len(backtesting_results), min_periods=1).apply(np.prod)
