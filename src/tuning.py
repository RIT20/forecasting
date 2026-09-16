import pickle
import pickle
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.data_prepare import fetch_prepared_data, id_generator
from utils.data_prepare import push_config_table_to_snowflake, create_config_data
from utils.io_utils import get_model_dir


# sys.path.insert(0, os.getcwd())


def get_tuned_params(df_h: pd.DataFrame, X, y):
    logger.info(f"Tuning Hyper Parameters")
    config = configuration.model.to_dict()
    tenant_id = getattr(DBManager, "tenant_id")

    try:

        def objective(trial):
            param_grid = {
                "n_estimators": trial.suggest_categorical("n_estimators", [2000]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 30, 70, step=5),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", 1, 10, step=1
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.7, 0.95, step=0.05
                ),
                "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            }

            horizon = config.get("horizon")
            eval1_dates = df_h["effective_date"].unique()[-horizon:]
            eval2_dates = df_h["effective_date"].unique()[-2 * horizon : -horizon]

            cv_scores = np.empty(2)
            for i in range(2):
                if i == 0:
                    eval_dates = eval1_dates
                else:
                    eval_dates = eval2_dates
                X_train = df_h[df_h.effective_date < eval_dates[0]][X]
                y_train = df_h[df_h.effective_date < eval_dates[0]][y[0]]
                X_test = df_h.query("effective_date in @eval_dates")[X]
                y_test = df_h.query("effective_date in @eval_dates")[y[0]]

                model = lgb.LGBMRegressor(**param_grid)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric="rmse",
                    early_stopping_rounds=100,
                    # callbacks=[
                    #     LightGBMPruningCallback(trial, "rmse")
                    # ],  # Add a pruning callback
                )
                preds = model.predict(X_test)
                cv_scores[i] = (y_test - preds).abs().sum() / y_test.abs().sum()

            return np.mean(cv_scores)

        study = optuna.create_study(
            direction="minimize", study_name=f"{tenant_id} Forecasting"
        )
        # func = lambda trial: objective(trial, X, y)
        study.optimize(objective, n_trials=config.get("tuning_n_trials"))

        return study.best_params, study.trials_dataframe()
    except Exception as e:
        logger.info("Exception in get_tuned_params ", e)


def save_data(cluster_id: int, params: dict, trials_df: pd.DataFrame, tune_run_id: int):
    logger.info(f"Export and save tuned hyper parameters")
    tenant_id = getattr(DBManager, "tenant_id")
    try:
        model_path = get_model_dir()
        location = "{}tuning/{}/{}".format(model_path, tenant_id, tune_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)

        with open(location + "/params_{}.pickle".format(cluster_id), "wb") as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)

        trials_df.to_csv(location + "/trials_info_{}.csv".format(cluster_id))
    except Exception as e:
        logger.info("Exception in save_data ", e)


def tune_for():
    try:
        t0 = time.time()

        tune_run_id = id_generator()

        logger.info(f"Fetching Data for tuning")
        df_h, X, y = fetch_prepared_data()

        for cluster_id in df_h.cluster_id.unique():
            df_h_c = df_h.query("cluster_id == @cluster_id")

            tuned_params, trials_df = get_tuned_params(df_h_c, X, y)
            save_data(cluster_id, tuned_params, trials_df, tune_run_id)

        logger.info(f"Save execution log")
        op_duration_sec = round(time.time() - t0, 3)
        config_df = create_config_data(tune_run_id, op_duration_sec)
        push_config_table_to_snowflake(config_df)

    except Exception as err:
        logger.error("Error in tune_for: %s", traceback.format_exc())
