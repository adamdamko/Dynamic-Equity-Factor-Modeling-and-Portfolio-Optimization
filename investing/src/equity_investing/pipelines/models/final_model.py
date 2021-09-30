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
#             - choose random_state *** for final model (lowest std in testing)


#%% Import libraries
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

#%% Get train data and splits
mod_data = pd.read_csv(
    'C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/modeling_data.csv'
)
X_train = pd.read_csv('C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_x_data.csv')
y_train = pd.read_csv('C:/Users/damko/PycharmProjects/Equity_Investing/investing/data/05_model_input/train_y_data.csv')
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
                                                                n_iter=500,
                                                                scoring='neg_mean_absolute_percentage_error',
                                                                cv=tscv,
                                                                verbose=10,
                                                                refit=True
                                                                ),
                                             transformer=StandardScaler()
                                             )
# Fit the estimator
lgbm_randomized.fit(X_train, y_train)  # categorical_feature is auto

#%% Compute metrics
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


#%% LIGHTGBM RANDOM SEED CONSISTENCY TESTING
# Using best hyperparameters from old tests
# LIGHTGBM PIPELINE
# Instantiate regressor
lgbm_seed = lgb.LGBMRegressor(n_estimators=145,
                              learning_rate=0.011,
                              reg_lambda=5.5,
                              num_leaves=5,
                              max_depth=205,
                              max_bin=130,
                              boosting_type='dart',
                              extra_trees=True,
                              n_jobs=23)  # bagging_freq=10 for future

# Create the parameter dictionary: params
lgbm_seed_param_grid = {
                        'lgbm_model__random_state': np.linspace(1000, 999999, 200, dtype=int)
                       }

# Setup the pipeline steps: steps
lgbm_seed_steps = [("lgbm_model", lgbm_seed)]

# Create the pipeline: xgb_pipeline
lgbm_seed_pipeline = Pipeline(lgbm_seed_steps)

# Perform random search: grid_mae
lgbm_seed_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_seed_pipeline,
                                                                     param_distributions=lgbm_seed_param_grid,
                                                                     n_iter=200,
                                                                     scoring='neg_mean_absolute_percentage_error',
                                                                     cv=tscv,
                                                                     verbose=10,
                                                                     refit=True
                                                                     ),
                                                  transformer=StandardScaler()
                                                  )
# Fit the estimator
lgbm_seed_randomized.fit(X_train, y_train)  # categorical_feature is auto

# Look at cv_results
lgbm_seed_cv_results = pd.DataFrame(lgbm_seed_randomized.regressor_.cv_results_)


#%% Save model cv's
# Initial model cv
# joblib.dump(lgbm_randomized, 'investing/data/06_models/lgbm_coarse_cv_model.pkl')
# 2nd model cv
# joblib.dump(lgbm_randomized, 'models/lgbm_2nd_cv.pkl')
# 3rd model cv
# joblib.dump(lgbm_randomized, 'models/lgbm_3rd_cv.pkl')

#%% Load model cv's
lgbm_2nd_cv = joblib.load('models/lgbm_2nd_cv.pkl')

#%% Look at model cv results
lgbm_2nd_cv_results = pd.DataFrame(lgbm_2nd_cv.regressor_.cv_results_)

#%%
testing = X_train.loc[X_train.index.isin(tscv[2]), :]