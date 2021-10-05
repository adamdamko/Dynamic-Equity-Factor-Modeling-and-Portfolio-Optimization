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
# from pandas_profiling import ProfileReport
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

#%% Get holdout data
hold_data = pd.read_csv(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/holdout_data.csv'
)

#%% Get X_test, y_test
x_test = hold_data.drop(columns=['date', 'ticker', 'target_1m_mom_lead'])
# Extract columns for StandardScaler
float_mask = (x_test.dtypes == 'float64')
# Get list of float column names
float_columns = x_test.columns[float_mask].tolist()
# Create StandardScaler object
scalerx = StandardScaler()
# Scale columns
x_test[float_columns] = scalerx.fit_transform(x_test[float_columns])
# Make as categorical
x_test.loc[:, 'sector'] = x_test.loc[:, 'sector'].astype('category')
x_test.loc[:, 'market_cap_cat'] = x_test.loc[:, 'market_cap_cat'].astype('category')
# Reset index to match with splits
x_test = x_test.reset_index(drop=True)
# Get y_test
y_test = hold_data['target_1m_mom_lead']

#%% Predict on x_test
test_x = X_train.filter(items=tscv[3][1], axis=0)
test_y = y_train.filter(items=tscv[3][1], axis=0)

#%%
# Get predictions just with predict
predictions = lgbm_fine_cv_model.predict(test_x)
# Get error
pred_error = mean_absolute_percentage_error(test_y, predictions)


#%% Get train data and splits

X_train = pd.read_csv('C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_x_data.csv')
y_train = pd.read_csv('C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_y_data.csv')
tscv = pd.read_pickle(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/time_series_split_list.pickle'
)

#%% EDA
# profile = ProfileReport(mod_data, title='Modeling Data Report')
# profile.to_file(
#     'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/model_data_eda.html')

#%% Make categorical vars as pd.Categorical
# Train X data
X_train.loc[:, 'sector'] = X_train.loc[:, 'sector'].astype('category')
X_train.loc[:, 'market_cap_cat'] = X_train.loc[:, 'market_cap_cat'].astype('category')

#%% Make dummy
# X_train = pd.get_dummies(X_train, columns=['sector', 'market_cap_cat'])

#%% LIGHTGBM
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm = lgb.LGBMRegressor(boosting_type='dart',
                         extra_trees=True,
                         # random_state=377507,  # lowest std random_state from testing
                         n_jobs=23)  # bagging_freq=10 for future

# Create the parameter dictionary: params
lgbm_param_grid = {'lgbm_model__n_estimators': np.linspace(50, 1750, 25, dtype=int),  # alias: num_iterations
                   'lgbm_model__learning_rate': np.round(np.linspace(0.01, 0.05, 25), 3),
                   'lgbm_model__max_depth': np.linspace(200, 750, 25, dtype=int),
                   'lgbm_model__reg_lambda': np.round(np.linspace(5, 20, 25), 3),  # alias: lambda_l2
                   'lgbm_model__num_leaves': np.linspace(2, 200, 25, dtype=int),
                   'lgbm_model__max_bin': np.linspace(30, 350, 25, dtype=int)
                   # 'lgbm_model__bagging_fraction': np.linspace(0.5, 1, 26)
                   # 'lgbm_model__min_data_in_leaf': np.linspace(2, 100, 50, dtype=int)
                   # 'lgbm_model__path_smooth': np.linspace(0, 10, 21)
                   }

# Setup the pipeline steps: steps
lgbm_steps = [("lgbm_model", lgbm)]

# Create the pipeline: xgb_pipeline
lgbm_pipeline = Pipeline(lgbm_steps)

# Perform random search: grid_mae
lgbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_pipeline,
                                                                param_distributions=lgbm_param_grid,
                                                                n_iter=50,
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
# lgbm_randomized = lgbm_fine_cv_model
# Print the best parameters and lowest MAE
print("Best estimators found: ", lgbm_randomized.regressor_.best_estimator_)
print("Best parameters found: ", lgbm_randomized.regressor_.best_params_)
print("Lowest MAPE found: ", np.abs(lgbm_randomized.regressor_.best_score_))
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
                  ylabel='MAPE',
                  title='MAPE for different {}s'.format(name))
    plt.gca().set_ylim([-4, 0])
    plt.show()


param_list = ['param_lgbm_model__n_estimators',
              'param_lgbm_model__learning_rate',
              'param_lgbm_model__max_depth',
              'param_lgbm_model__reg_lambda',
              'param_lgbm_model__num_leaves',
              'param_lgbm_model__max_bin']


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
lgbm_seed = lgb.LGBMRegressor(n_estimators=750,
                              learning_rate=0.012,
                              reg_lambda=2,
                              num_leaves=127,
                              # max_depth=350,
                              max_bin=460,
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
                                          n_iter=100,
                                          scoring='neg_mean_absolute_percentage_error',
                                          cv=tscv,
                                          verbose=10,
                                          refit=True
                                          )
# Fit the estimator
lgbm_seed_randomized.fit(X_train, y_train)  # categorical_feature is auto

# Look at cv_results
lgbm_seed_cv_results = pd.DataFrame(lgbm_seed_randomized.cv_results_)

#%% Testing
# Transform y
# y_train = pd.DataFrame(scalerx.fit_transform(y_train))
# Set testers
train_x = X_train.filter(items=tscv[3][0], axis=0)
train_y = y_train.filter(items=tscv[3][0], axis=0)

lgb_test = TransformedTargetRegressor(lgb.LGBMRegressor(n_estimators=1000,  # 140,
                                                        learning_rate=0.012,
                                                        reg_lambda=2,  # 10.625,
                                                        num_leaves=127,  # 6,
                                                        # max_depth=10,  #350,
                                                        max_bin=460,
                                                        boosting_type='dart',
                                                        # extra_trees=True,
                                                        n_jobs=23),
                                      transformer=StandardScaler()
                                      )

# Fit the estimator
lgb_test.fit(train_x, train_y)

# Test preds
test_x = X_train.filter(items=tscv[3][1], axis=0)
test_y = pd.DataFrame(y_train.filter(items=tscv[3][1], axis=0)).reset_index(drop=True)
test_preds = pd.DataFrame(lgb_test.predict(test_x))
# Get mse
test_error = mean_squared_error(test_y, test_preds)
# Join preds and test
test_view = pd.concat([test_y, test_preds], axis=1)

#%%
test_view_2 = test_view.sort_values(by=0, ascending=False).head(100)

#%% Save model cv's
# Initial model cv
# joblib.dump(lgbm_randomized, 'investing/data/06_models/lgbm_coarse_cv_model.pkl')
# 2nd model cv
# joblib.dump(lgbm_randomized, 'models_ml_final/lgbm_2nd_cv.pkl')
# 3rd model cv
# joblib.dump(lgbm_randomized, 'models_ml_final/lgbm_3rd_cv.pkl')

#%% Load model cv's
lgbm_fine_cv_model = joblib.load(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/06_models/lgbm_coarse_cv_model.pkl'
)

# pd.read_pickle(
#     'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/time_series_split_list.pickle'
# )


#%% SHAP
# explainer = shap.TreeExplainer(lgbm_fine_cv_model)
# shap_values = explainer.shap_values(X_train)

# shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_train.iloc[0, :])
