# from optuna.integration import LightGBMPruningCallback

import os
import sys
sys.path.insert(0, os.getcwd())

import traceback
import pickle
from pathlib import Path
import math
import optuna 
import lightgbm as lgb
import numpy as np
import time

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger

from utils.data_prepare_do_bm import id_generator,get_Xny
from utils.data_prepare_do_bm import push_config_table_to_snowflake, create_config_data
from utils.io_utils import get_model_dir, get_root_dir

def get_tuned_params(config, df_h, X, y):
    try:
        start_time = time.time()
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
            # eval2_dates = df_h['effective_date'].unique()[-2*horizon:-horizon]

            cv_scores = np.empty(1)
            for i in range(1):
                eval_dates = eval1_dates
                X_train = df_h[df_h.effective_date < eval_dates[0]][X]
                y_train = df_h[df_h.effective_date < eval_dates[0]][y[0]]
                X_test = df_h.query('effective_date in @eval_dates')[X]
                y_test = df_h.query('effective_date in @eval_dates')[y[0]]
                # model = lgb.LGBMRegressor(**param_grid)
                model = lgb.LGBMClassifier(**param_grid)

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric="rmse",
                    early_stopping_rounds=30,
                    # callbacks=[
                    #     LightGBMPruningCallback(trial, "rmse")
                    # ],  # Add a pruning callback
                )
                preds = model.predict(X_test)
                cv_scores[i] = (y_test - preds).abs().sum() / y_test.abs().sum()
            if math.isnan(np.mean(cv_scores)):
                return 1e100
            return np.mean(cv_scores)

        study = optuna.create_study(direction="minimize", study_name="Ecom Forecasting")
        # func = lambda trial: objective(trial, X, y)
        study.optimize(objective, n_trials=config.get('tuning_n_trials'))
        with open("output_do.txt", "a") as f:
            print(f"get_tuned_params completed in {(time.time() - start_time)} seconds", file=f)
        return study.best_params, study.trials_dataframe()
    except Exception as e :
        logger.error(f'Exeption in get_tuned_params:{e} ')

    

def save_data(tenet,cluster_id, params, trials_df, tune_run_id):
    try:    
        model_path = get_root_dir()
        # model_path = CONFIG_TABLES['model_location']
        location = '{}/tuning_do/{}/{}'.format(model_path, tenet, tune_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)

        with open(location + '/params_{}.pickle'.format(cluster_id), 'wb') as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

        trials_df.to_csv(location + '/trials_info_{}.csv'.format(cluster_id))
    except Exception as e :
        logger.error(f'Exeption in save_data:{e} ')


def tune_for(config,tune_run_id, df_h):
    print('Tune for called')
    try:
        logger.info ("Tuning --started")
        start_time = time.time()
        tenant_id = config['tenant_id']
        tune_run_id=tune_run_id
        X, y = get_Xny(config)
        cluster_run_id = 1
        # df_h, X, y,cluster_run_id  = fetch_prepared_data(config, conn, df)
        config['cluster_run_id'] = cluster_run_id
        # Added this line
        df_h["cluster_id"] = 1
        for cluster_id in df_h.cluster_id.unique():
            df_h_c = df_h.query('cluster_id == @cluster_id')
            logger.info("get_tuned_params --started")
            tuned_params, trials_df = get_tuned_params(config, df_h_c, X, y)
            logger.info("get_tuned_params --completed")
            save_data(tenant_id, cluster_id, tuned_params, trials_df, tune_run_id)
            logger.info("save tune data --completed")
        # config_df = create_config_data(config, tune_run_id, op_duration_sec)
        # push_config_table_to_snowflake(config, config_df, conn)
        with open("output_do.txt", "a") as f:
            print(f"tune_for completed in {(time.time() - start_time)} seconds", file=f)
        logger.info("Tuning --completed")
    except Exception as e :
        logger.error(f"Error in tune_for: {e}")
