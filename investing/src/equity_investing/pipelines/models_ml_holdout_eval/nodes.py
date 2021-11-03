# This is the models_ml_holdout eval nodes file.

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


class HoldoutValidation:
    """
    The HoldoutValidation class trains and tests the final model in the exact same
    manner as the productionized model.

    """
    @staticmethod
    def validation(modeling_data: pd.DataFrame, validation_train_dates_list: Dict,
                   validation_test_dates_list: Dict, lgbm_fine_best_params: Dict,
                   model_features: Dict, model_target: Dict) -> pd.DataFrame:
        """
        This function creates the target train/test variable.

        Args:
            modeling_data: Output from 'feature_engineering' pipeline.
            validation_train_dates_list: Monthly dates to loop through for validation. Train on everything less
                                         than each date and predict on next date's data.
                                         Use 'LightGBM best fine-tuning parameters' parameter.
                                         This replicates real-world use.
            validation_test_dates_list: Test dates for validation.
            lgbm_fine_best_params: Best-tuned parameters from LightGBM hyperparameter tuning.
            model_features: Final feature set for modeling.
            model_target: Final target variable for modeling.

        Returns:
             Pandas dataframe
        """
        # RESULTS TO APPEND
        results = []

        for train, test in list(zip(validation_train_dates_list, validation_test_dates_list)):

            # CREATE X_TRAIN AND Y_TRAIN
            data_2 = modeling_data.reset_index(drop=True)
            data_2 = data_2[data_2['date'] < train]
            data_2 = data_2.reset_index(drop=True)
            # Set categorical variables as 'category'
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
            # Set variables for data sets
            y_train = data_2[model_target]
            x_train = x_train[model_features]

            # CREATE X_TEST AND Y_TEST
            data_3 = modeling_data.reset_index(drop=True)
            data_3 = data_3[(data_3['date'] >= train) & (data_3['date'] < test)]
            data_3 = data_3.reset_index(drop=True)
            data_3 = data_3.set_index(['date', 'ticker'])
            # Set categorical variables as 'category'
            data_3.loc[:, 'sector'] = data_3.loc[:, 'sector'].astype('category')
            data_3.loc[:, 'market_cap_cat'] = data_3.loc[:, 'market_cap_cat'].astype('category')
            # Extract columns for StandardScaler
            x_test = data_3.copy()
            float_mask = (x_test.dtypes == 'float64')
            # Get list of float column names
            float_columns = x_test.columns[float_mask].tolist()
            # Scale columns
            x_test[float_columns] = scalerx.fit_transform(x_test[float_columns])
            # Set variables for data sets
            y_test = pd.DataFrame(data_3[model_target])
            x_test = x_test[model_features]

            # LIGHTGBM MODEL
            lgb_model = TransformedTargetRegressor(lgb.LGBMRegressor(
                                                                n_estimators=lgbm_fine_best_params['n_estimators'],
                                                                learning_rate=lgbm_fine_best_params['learning_rate'],
                                                                max_depth=lgbm_fine_best_params['max_depth'],
                                                                num_leaves=lgbm_fine_best_params['num_leaves'],
                                                                min_data_in_leaf=lgbm_fine_best_params['num_leaves'],
                                                                boosting_type=lgbm_fine_best_params['boosting_type'],
                                                                n_jobs=lgbm_fine_best_params['n_jobs'],
                                                                random_state=lgbm_fine_best_params['random_state']
            ),
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

        # Make results as pandas dataframe
        final_results = pd.DataFrame()
        for result in results:
            final_results = pd.concat([final_results, result], axis=0)

        # Rename preds column
        final_results = final_results.rename(columns={0: 'predictions'})

        return final_results

    @staticmethod
    def top_predictions_view(holdout_results_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the target train/test variable.

        Args:
            holdout_results_data: Results of holdout data testing.

        Returns:
             Pandas dataframe of top 100 predictions per month only.
        """
        # Calculate summary output
        final_results2 = holdout_results_data.reset_index()
        # Get top 100 predictions per month only view
        top_preds_returns = final_results2.sort_values(['date', 'predictions'], ascending=False)
        top_preds = top_preds_returns.groupby('date')['predictions'].nlargest(100).reset_index().set_index('level_1')
        top_preds_returns = pd.merge(top_preds, top_preds_returns, how='left', left_index=True, right_index=True)
        top_preds_returns = top_preds_returns.drop(columns=['date_x', 'predictions_x']). \
            rename(columns={'date_y': 'date', 'predictions_y': 'predictions'})

        return top_preds_returns

    @staticmethod
    def performance_summary(holdout_results_data: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates the target train/test variable.

        Args:
            holdout_results_data: Results of holdout data testing.

        Returns:
             Pandas dataframe of performance summary of market and top predictions over
             the holdout period.
        """
        # Calculate summary output
        final_results = holdout_results_data.reset_index()
        # Get top 100 predictions per month only view
        top_preds_returns = final_results.sort_values(['date', 'predictions'], ascending=False)
        top_preds = top_preds_returns.groupby('date')['predictions'].nlargest(100).reset_index().set_index('level_1')
        top_preds_returns = pd.merge(top_preds, top_preds_returns, how='left', left_index=True, right_index=True)
        top_preds_returns = top_preds_returns.drop(columns=['date_x', 'predictions_x']). \
            rename(columns={'date_y': 'date', 'predictions_y': 'predictions'})
        # Join returns
        top_preds_returns2 = top_preds_returns.groupby('date').agg({'target_1m_mom_lead': 'mean'}). \
            rename(columns={'target_1m_mom_lead': 'top_preds_returns'})
        total_market_returns = final_results.groupby('date').agg({'target_1m_mom_lead': 'mean'})
        returns_comp = pd.merge(total_market_returns, top_preds_returns2, how='left', left_index=True, right_index=True)
        returns_comp.loc[:, 'market_rolling_return'] = returns_comp['target_1m_mom_lead'].rolling(window=19,
                                                                                                  min_periods=1). \
            apply(np.prod)
        returns_comp.loc[:, 'top_preds_rolling_return'] = returns_comp['top_preds_returns'].rolling(window=19,
                                                                                                    min_periods=1). \
            apply(np.prod)

        return returns_comp
