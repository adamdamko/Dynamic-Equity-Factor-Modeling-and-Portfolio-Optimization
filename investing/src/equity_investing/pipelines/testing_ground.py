# This is the chosen modeling method (Lightgbm) file.
# Comments on hyperparameters from initial 'coarse' tuning:
#   n_estimators (num_iterations) : best performance n_estimators < 300
#   learning_rate: best performance at 0.01 < learning_rate < 0.02
#             - **Tradeoff between learning rate and n_estimators. Smaller learning_rate
#               seems to improve performance (<0.01) but requires more n_estimators to
#               to reach convergence. But performance improves greatly with fewer
#               n_estimators (150 < n_estimators < 300). Convergence issues at learning_rate
#               (<0.01). So learning_rate at 0.01 was minimum used.**
#   max_depth: performance does not vary much from lower to higher levels
#             - keep this value lower to reduce over-fitting
#   reg_lambda (lambda_l2): performance does not vary much between values
#             - use higher value to reduce over-fitting
#   num_leaves: performance decreases as num_leaves gets larger
#             - num_leaves <= 10 seems best
#             - small number of leaves also reduces over-fitting
#   max_bin: performance does not vary much between values
#             - keep low to reduce over-fitting
#   extra_trees: used to reduce over-fitting
#   dart: seems to improve performance
#   random_state: after testing, random_state is relatively stable
#             - choose random_state 126502 for final model (lowest std in testing)


#%% Import libraries
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


#%% Get train data and splits
X_train = pd.read_csv('/investing/data/05_model_input/train_x_data.csv')
y_train = pd.read_csv('/investing/data/05_model_input/train_y_data.csv')
tscv = pd.read_pickle(
    '/investing/data/05_model_input/time_series_split_list.pickle'
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


#%% Visualize hyperparameters


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

#%%
fig, axes = plt.subplots(6)
for ax, param in zip(axes, param_list):
    ax.scatter(lgbm_cv_results[param], lgbm_cv_results['mean_test_score'], c=['blue'])
    ax.set(xlabel='{}'.format(param),
           ylabel='MAPE',
           title='MAPE for different {}s'.format(param))


plt.show()
#%% LIGHTGBM RANDOM SEED CONSISTENCY TESTING
# Using best hyperparameters from old tests
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm_seed = lgb.LGBMRegressor(n_estimators=122,
                              learning_rate=0.01,
                              reg_lambda=14.167,
                              num_leaves=75,
                              min_data_in_leaf=259,
                              boosting_type='dart',
                              extra_trees=True,
                              n_jobs=23)

# Create the parameter dictionary: params
lgbm_seed_param_grid = {
                        'lgbm_model__random_state': np.linspace(1000, 999999, 200, dtype=int)
                       }

# Setup the pipeline steps: steps
lgbm_seed_steps = [("lgbm_model", lgbm_seed)]

# Create the pipeline: xgb_pipeline
lgbm_seed_pipeline = Pipeline(lgbm_seed_steps)

# Perform random search: grid_mae
lgbm_seed_randomized = RandomizedSearchCV(estimator=lgbm_seed_pipeline,
                                          param_distributions=lgbm_seed_param_grid,
                                          n_iter=200,
                                          scoring='neg_root_mean_squared_error',
                                          cv=tscv,
                                          verbose=10,
                                          refit=True
                                          )
# Fit the estimator
lgbm_seed_randomized.fit(X_train, y_train)  # categorical_feature is auto

# Look at cv_results
lgbm_seed_cv_results = pd.DataFrame(lgbm_seed_randomized.cv_results_)


#%% Load model cv's
modeling_data = pd.read_parquet(
    '/investing/data/04_feature/modeling_data.csv'
)

#%% Holdout eval testing
validation_train_dates_list = [ '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01',
                               '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01',
                               '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01',
                               '2021-08-01']

# Test dates
validation_test_dates_list = [ '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
                              '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01',
                              '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01',
                              '2021-09-01' ]

results = []

for train, test in list(zip(validation_train_dates_list, validation_test_dates_list)):
    # CREATE X_TRAIN AND Y_TRAIN
    data_2 = modeling_data.reset_index(drop=True)
    data_2 = data_2[data_2['date'] < train]
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
    y_train = data_2['target_1m_mom_lead']
    x_train = x_train.drop(columns=['date', 'ticker', 'target_1m_mom_lead'])

    # CREATE X_TEST AND Y_TEST
    data_3 = modeling_data.reset_index(drop=True)
    data_3 = data_3[(data_3['date'] >= train) & (data_3['date'] < test)]
    data_3 = data_3.reset_index(drop=True)
    data_3 = data_3.set_index(['date', 'ticker'])
    # Train X data
    data_3.loc[:, 'sector'] = data_3.loc[:, 'sector'].astype('category')
    data_3.loc[:, 'market_cap_cat'] = data_3.loc[:, 'market_cap_cat'].astype('category')
    # Extract columns for StandardScaler
    x_test = data_3.copy()
    float_mask = (x_test.dtypes == 'float64')
    # Get list of float column names
    float_columns = x_test.columns[float_mask].tolist()
    # Scale columns
    x_test[float_columns] = scalerx.fit_transform(x_test[float_columns])
    x_test = x_test.drop(columns=['target_1m_mom_lead'])
    y_test = pd.DataFrame(data_3['target_1m_mom_lead'])

    # LIGHTGBM MODEL
    lgb_model = TransformedTargetRegressor(lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.0006,
        max_depth=7,
        num_leaves=80,
        min_data_in_leaf=5000,
        boosting_type='dart',
        n_jobs=23,
        random_state=638552),
        transformer=StandardScaler()
    )

    # Fit the estimator
    lgb_model.fit(x_train, y_train)

    # Test preds
    model_preds = pd.DataFrame(lgb_model.predict(x_test))

    # Join model_preds and y_test
    test_view = pd.concat([y_test, model_preds.set_index(y_test.index)], axis=1)

    # Append results
    results.append(test_view)

# Make results into dataframe
final_results = pd.DataFrame()
for result in results:
    final_results = pd.concat([final_results, result], axis=0)

# Rename preds column
final_results = final_results.rename(columns={0: 'predictions'})


#%%
holdout_data = pd.read_parquet(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/07_model_output/holdout_results_data.parquet'
)

#%% Calculate summary output
final_results2 = final_results.reset_index()

top_preds_returns = final_results2.sort_values(['date', 'predictions'], ascending=False)
top_preds = top_preds_returns.groupby('date')['predictions'].nlargest(100).reset_index().set_index('level_1')
top_preds_returns = pd.merge(top_preds, top_preds_returns, how='left', left_index=True, right_index=True)
top_preds_returns = top_preds_returns.drop(columns=['date_x', 'predictions_x']).\
    rename(columns={'date_y': 'date', 'predictions_y': 'predictions'})

# Join returns
top_preds_returns2 = top_preds_returns.groupby('date').agg({'target_1m_mom_lead': 'mean'}).\
    rename(columns={'target_1m_mom_lead': 'top_preds_returns'})
total_market_returns = final_results2.groupby('date').agg({'target_1m_mom_lead': 'mean'})
returns_comp = pd.merge(total_market_returns, top_preds_returns2, how='left', left_index=True, right_index=True)
returns_comp.loc[:, 'market_rolling_return'] = returns_comp['target_1m_mom_lead'].rolling(window=19, min_periods=1).\
    apply(np.prod)
returns_comp.loc[:, 'top_preds_rolling_return'] = returns_comp['top_preds_returns'].rolling(window=19, min_periods=1).\
    apply(np.prod)


#%%
preds_tickers = top_preds_returns[top_preds_returns['date'] == '2021-02-26']
preds_tickers2 = top_preds_returns[top_preds_returns['date'] == '2021-04-30']
print(preds_tickers[preds_tickers['ticker'].isin(list(preds_tickers2['ticker']))])
