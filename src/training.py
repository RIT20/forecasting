import pickle
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.data_prepare import fetch_prepared_data, id_generator
from utils.data_prepare import push_config_table_to_snowflake, create_config_data
from utils.io_utils import get_model_dir


# sys.path.insert(0, os.getcwd())


def get_n_estimators(df, X, y, horizon, params):
    try:
        eval_cutoff = df["effective_date"].unique()[-horizon:][0]
        train = df.query("effective_date < @eval_cutoff")
        eval_set = df.query("effective_date >= @eval_cutoff")

        model = lgb.LGBMRegressor(**params)
        model.fit(
            train[X],
            train[y],
            eval_set=[(eval_set[X], eval_set[y])],
            eval_metric="rmse",
            early_stopping_rounds=100,
            verbose=True,
        )

        return model.best_iteration_
    except Exception as e:
        logger.info("Exception in get_n_estimators ", e)


def get_params(tune_run_id, cluster_id):
    tenant_id = getattr(DBManager, "tenant_id")

    try:
        model_path = get_model_dir()
        location = "{}tuning/{}/{}".format(model_path, tenant_id, tune_run_id)
        with open(location + "/params_{}.pickle".format(cluster_id), "rb") as file:
            params = pickle.load(file)

        return params
    except Exception as e:
        logger.info("Exception in get_params ", e)


def build_model(df_h, X, y, tune_run_id, cluster_id, quantile):
    try:
        horizon = getattr(configuration.model, "horizon")

        train = df_h[X + y]
        params = get_params(tune_run_id, cluster_id)

        if quantile == 0.5:
            params['objective'] = 'regression'
        else:
            params['objective'] = 'quantile'
            params['metric'] = 'qunatile'
            params['alpha'] = quantile

        n_est = get_n_estimators(df_h, X, y, horizon, params)
        params["n_estimators"] = np.max([400, n_est])

        model = lgb.LGBMRegressor(**params)
        model.fit(train[X], train[y], verbose=True)

        return model
    except Exception as e:
        logger.info("Exception in build_model ", e)


def save_model(model, cluster_id, train_run_id, quantile):
    tenant_id = getattr(DBManager, "tenant_id")

    try:
        model_path = get_model_dir()
        location = "{}training/{}/{}".format(model_path, tenant_id, train_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(
            location + "/model_{}_{}.txt".format(cluster_id, quantile)
        )
    except Exception as e:
        logger.info("Exception in save_model ", e)


def get_tune_runid_from_snowflake():
    conn = getattr(DBManager, "engine")
    tenant_id = getattr(DBManager, "tenant_id")

    table_name = getattr(configuration.orm, "ForecastMetaDataTable")

    qry = """select op_run_id 
            from  {table}
            where   updated_on = (select max(updated_on) 
                                from {table}
                                where op_name = 'tuning')
                    and tenant_id = '{tenant_id}'
                    and op_name = 'tuning'"""
    qry_format = qry.format(tenant_id=tenant_id, table=table_name)

    tune_runid = conn.execute(qry_format).fetchone()[0]
    return tune_runid


def train_for():
    try:
        t0 = time.time()

        train_run_id = id_generator()
        tune_run_id = get_tune_runid_from_snowflake()

        logger.info(f"Fetching Data for training")
        df_h, X, y = fetch_prepared_data()

        logger.info(f"Training Model")
        for cluster_id in df_h.cluster_id.unique():
            try:
                for quantile in [0.05, 0.5, 0.95]:
                    df_h_c = df_h.query("cluster_id == @cluster_id")
                    model = build_model(df_h_c, X, y, tune_run_id, cluster_id, quantile)
                    save_model(model, cluster_id, train_run_id, quantile)
            except Exception as e:
                logger.info("Could not train for cluster: {}".format(cluster_id), e)
                continue

        logger.info(f"Save execution log")
        op_duration_sec = round(time.time() - t0, 3)
        config_df = create_config_data(train_run_id, op_duration_sec, tune_run_id)
        push_config_table_to_snowflake(config_df)
    except Exception as err:
        logger.error("Error in train_for: %s", traceback.format_exc())
