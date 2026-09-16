# from optuna.integration import LightGBMPruningCallback

import os
import sys

sys.path.insert(0, os.getcwd())
import math
import traceback
import logging
import logging.config
import pickle
from pathlib import Path

import optuna
import lightgbm as lgb
import numpy as np
import time
from utils.data_prepare_regression_bm import id_generator,get_Xny
from utils.data_prepare_regression_bm import push_config_table_to_snowflake, create_config_data

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.io_utils import get_model_dir,get_root_dir

def get_tuned_params(config, df_h, X, y):
    start_time = time.time()
    try:

        def objective(trial):
            param_grid = {
                "n_estimators": trial.suggest_categorical("n_estimators", [2000]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 30, 70, step=5),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10, step=1),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.7, 0.95, step=0.05
                ),
                "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            }

            horizon = config.get('horizon')
            eval1_dates = sorted(df_h['effective_date'].unique())[-horizon:]
            eval2_dates = sorted(df_h['effective_date'].unique())[-2 * horizon:-horizon]
            # import pdb;pdb.set_trace()
            cv_scores = np.empty(2)
            for i in range(2):
                if i == 0:
                    eval_dates = eval1_dates
                else:
                    eval_dates = eval2_dates
                X_train = df_h[df_h.effective_date < eval_dates[0]][X]
                y_train = df_h[df_h.effective_date < eval_dates[0]][y[0]]
                X_test = df_h.query('effective_date in @eval_dates')[X]
                y_test = df_h.query('effective_date in @eval_dates')[y[0]]

                model = lgb.LGBMRegressor(**param_grid)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric="rmse",
                    early_stopping_rounds=10,
                    # callbacks=[
                    #     LightGBMPruningCallback(trial, "rmse")
                    # ],  # Add a pruning callback
                )
                preds = model.predict(X_test)
                cv_scores[i] = (y_test - preds).abs().sum() / y_test.abs().sum()
            if math.isnan(np.mean(cv_scores)):
                return 1e100
            return np.mean(cv_scores)

        study = optuna.create_study(direction="minimize", study_name="Bashas Forecasting")
        # func = lambda trial: objective(trial, X, y)
        study.optimize(objective, n_trials=config.get('tuning_n_trials'))
        with open("output.txt", "a") as f:
            print(f"get_tuned_params completed in {(time.time() - start_time)} seconds", file=f)
        return study.best_params, study.trials_dataframe()
    except Exception as e:
        logger.error(f'Exeption in get_tuned_params:{e} ')



def save_data(tenet, cluster_id, params, trials_df, tune_run_id):
    try:
        model_path = get_root_dir()
        location = '{}/tuning_regression/{}/{}'.format(model_path, tenet, tune_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)

        with open(location + '/params_{}.pickle'.format(cluster_id), 'wb') as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

        trials_df.to_csv(location + '/trials_info_{}.csv'.format(cluster_id))
    except Exception as e:
        logger.error(f'Exeption in save_data:{e} ')


def tune_for(config, df_h, tune_run_id):
    start_time = time.time()
    try:
        conn = getattr(DBManager, "engine")
        tenant_id = config['tenant_id']
        cluster_run_id = 1
        X, y = get_Xny(config)
        logger.info("fetch data for tuning --completed")
        config['cluster_run_id'] = cluster_run_id

        for cluster_id in df_h.cluster_id.unique():
            df_h_c = df_h.query('cluster_id == @cluster_id')
            tuned_params, trials_df = get_tuned_params(config, df_h_c, X, y)
            save_data(tenant_id, cluster_id, tuned_params, trials_df, tune_run_id)

    except Exception as e:
        logger.error(f"Error in tune_for: {e}")
