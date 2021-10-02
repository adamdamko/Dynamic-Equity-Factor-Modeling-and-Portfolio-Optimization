# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from equity_investing.pipelines import data_processing as dp
from equity_investing.pipelines import feature_engineering as fe
from equity_investing.pipelines import models_ml_exploratory as me
from equity_investing.pipelines import models_ml_final as mf


# Run the registry
def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_processing_pipeline = dp.create_pipeline()
    raw_data_eda_pipeline = dp.create_eda_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    final_data_eda_pipeline = fe.create_final_eda_pipeline()
    train_test_split_pipeline = mf.create_test_train_validation_sets_pipeline()
    exploratory_models_pipeline = me.create_exploratory_models_pipeline()
    coarse_hyperparameter_tuning_pipeline = mf.create_coarse_hyperparameter_tuning_pipeline()
    fine_hyperparameter_tuning_pipeline = mf.create_fine_hyperparameter_tuning_pipeline()

    return {
        # Individual pipelines
        "data_processing": data_processing_pipeline,
        "raw_data_eda": raw_data_eda_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "final_data_eda": final_data_eda_pipeline,
        "train_test_splits": train_test_split_pipeline,
        "exploratory_models": exploratory_models_pipeline,
        "coarse_hyperparameter_tuning": coarse_hyperparameter_tuning_pipeline,
        "fine_hyperparameter_tuning": fine_hyperparameter_tuning_pipeline,


        # PIPELINES FOR EXECUTION
        # Default pipeline
        "__default__": data_processing_pipeline + feature_engineering_pipeline +
                       train_test_split_pipeline + fine_hyperparameter_tuning_pipeline,
        # EDA pipelines
        "raw_eda": data_processing_pipeline + raw_data_eda_pipeline,
        "full_eda": data_processing_pipeline + raw_data_eda_pipeline + feature_engineering_pipeline +
                    final_data_eda_pipeline,
        # Full pipeline (takes long to run)
        "full_pipeline_ex_eda": data_processing_pipeline + feature_engineering_pipeline +
                                train_test_split_pipeline + exploratory_models_pipeline +
                                coarse_hyperparameter_tuning_pipeline + fine_hyperparameter_tuning_pipeline,
    }
