# This file is the testing ground file for different modeling
# techniques. Three techniques were tried: Random Forest, XGBoost,
# and Lightgbm. In initial testing Lightgbm showed slightly better
# performance in cross-validation and was much faster. Therefore,
# Lightgbm was chosen for the final modeling technique. To play with
# the below models again dummy variables will have to be created
# for the data sets as Lightgbm has it's own handling of categorical
# variables and the data sets were changed to not create dummy
# variables.


#%% Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


#%% RANDOM FOREST
# RANDOM FOREST PIPELINE
# Instantiate model
rfr = RandomForestRegressor(n_jobs=23)

# Create the parameter dictionary: params
rfr_param_grid = {'rfr_model__n_estimators': np.arange(250, 1250, 250),
                  'rfr_model__max_depth': np.arange(2, 12, 2),
                  'rfr_model__min_samples_split': np.arange(2, 16, 2),
                  'rfr_model__min_samples_leaf': np.arange(1, 10, 1)
                  }

# Setup the pipeline steps: steps
rfr_steps = [("rfr_model", rfr)]

# Create the pipeline: rfr_pipeline
rfr_pipeline = Pipeline(rfr_steps)

# Build a random search using param_dist and rfr
rfr_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=rfr_pipeline,
                                                               param_distributions=rfr_param_grid,
                                                               n_iter=50,
                                                               scoring='neg_mean_absolute_percentage_error',
                                                               cv=tscv,
                                                               verbose=2,
                                                               refit=True
                                                               ),
                                            transformer=StandardScaler()
                                            )

# Fit the estimator
rfr_randomized.fit(X_train, y_train)

#%% Compute metrics
# Print the best parameters and lowest MAE
print("Best estimators found: ", rfr_randomized.regressor_.best_estimator_)
print("Best parameters found: ", rfr_randomized.regressor_.best_params_)
print("Lowest MAPE found: ", np.abs(rfr_randomized.regressor_.best_score_))


#%% XGBOOST
# XGBOOST PIPELINE
# Instantiate regressor
gbm = xgb.XGBRegressor(tree_method='hist',
                       booster='gbtree',
                       )

# Create the parameter dictionary: params
gbm_param_grid = {'xgb_model__n_estimators': np.arange(250, 1250, 250),
                  'xgb_model__eta': np.arange(0.1, 1.0, 0.1),
                  'xgb_model__gamma': np.arange(0, 5, 0.5),
                  'xgb_model__max_depth': np.arange(2, 12, 2),
                  'xgb_model__subsample': np.arange(0.5, 1, 0.1),
                  'xgb_model__lambda': np.arange(1, 51, 5),
                  'xgb_model__colsample_bytree': np.arange(0.5, 1, 0.1)
                  }

# Setup the pipeline steps: steps
gbm_steps = [("xgb_model", gbm)]

# Create the pipeline: xgb_pipeline
gbm_pipeline = Pipeline(gbm_steps)

# Perform random search: grid_mae
gbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=gbm_pipeline,
                                                               param_distributions=gbm_param_grid,
                                                               n_iter=50,
                                                               scoring='neg_mean_absolute_percentage_error',
                                                               cv=tscv,
                                                               verbose=2,
                                                               n_jobs=22,
                                                               refit=True
                                                               ),
                                            transformer=StandardScaler()
                                            )
# Fit the estimator
gbm_randomized.fit(X_train, y_train)

#%% Compute metrics
# Print the best parameters and lowest MAE
print("Best estimators found: ", gbm_randomized.regressor_.best_estimator_)
print("Best parameters found: ", gbm_randomized.regressor_.best_params_)
print("Lowest MAPE found: ", np.abs(gbm_randomized.regressor_.best_score_))
# Get feature importance
xgb.plot_importance(gbm_randomized.regressor_.best_estimator_.named_steps['xgb_model'], max_num_features=40)
plt.show()
# Look at cv_results
gbm_cv_results = pd.DataFrame(gbm_randomized.regressor_.cv_results_)


#%% LIGHTGBM
# # LIGHTGBM PIPELINE
# # Instantiate regressor
# lgbm = lgb.LGBMRegressor(boosting_type='dart',
#                          extra_trees=True,
#                          n_jobs=-1)
#
# # Create the parameter dictionary: params
# lgbm_param_grid = {'lgbm_model__n_estimators': np.arange(500, 2500, 10),
#                    'lgbm_model__learning_rate': np.arange(0.01, 0.5, 0.01),
#                    'lgbm_model__num_leaves': np.arange(5, 60, 1),
#                    'lgbm_model__max_depth': np.arange(5, 200, 1),
#                    'lgbm_model__min_child_samples': np.arange(0, 100, 1),
#                    'lgbm_model__colsample_bytree': np.arange(0.5, 1, 0.025),
#                    'lgbm_model__reg_lambda': np.arange(0, 5, 0.02)
#                    }
#
# # Setup the pipeline steps: steps
# lgbm_steps = [("lgbm_model", lgbm)]
#
# # Create the pipeline: xgb_pipeline
# lgbm_pipeline = Pipeline(lgbm_steps)
#
# # Perform random search: grid_mae
# lgbm_randomized = TransformedTargetRegressor(RandomizedSearchCV(estimator=lgbm_pipeline,
#                                                                 param_distributions=lgbm_param_grid,
#                                                                 n_iter=500,
#                                                                 scoring='neg_mean_absolute_percentage_error',
#                                                                 cv=tscv,
#                                                                 verbose=2,
#                                                                 refit=True
#                                                                 ),
#                                              transformer=StandardScaler()
#                                              )
# # Fit the estimator
# lgbm_randomized.fit(X_train, y_train)
#
# #%% Compute metrics
# # Print the best parameters and lowest MAE
# print("Best estimators found: ", lgbm_randomized.regressor_.best_estimator_)
# print("Best parameters found: ", lgbm_randomized.regressor_.best_params_)
# print("Lowest MAPE found: ", np.abs(lgbm_randomized.regressor_.best_score_))
# # Get feature importance
# lgb.plot_importance(lgbm_randomized.regressor_.best_estimator_.named_steps['lgbm_model'], max_num_features=40)
# plt.show()
# # Look at cv_results
# lgbm_cv_results = pd.DataFrame(lgbm_randomized.regressor_.cv_results_)