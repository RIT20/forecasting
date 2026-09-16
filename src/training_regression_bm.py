import os
import sys

sys.path.insert(0, os.getcwd())
import traceback

from pathlib import Path
import pickle

import lightgbm as lgb
import lightgbm as lgb
import numpy as np
import time
from utils.data_prepare_regression_bm import id_generator
from utils.data_prepare_regression_bm import push_config_table_to_snowflake, create_config_data
from utils.data_prepare_regression_bm import get_Xny

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.io_utils import get_model_dir,get_root_dir

def get_n_estimators(df, X, y, horizon, params):
    try:
        horizon = 4
        eval_cutoff = sorted(df['effective_date'].unique())[-horizon:][0]
        print(eval_cutoff)
        train = df.query('effective_date < @eval_cutoff')
        print('###################')
        print(train.shape[1])
        print(train.head())
        eval_set = df.query('effective_date >= @eval_cutoff')
        print('###################eval_set')
        print(eval_set.head())
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train[X],
            train[y],
            eval_set=[(eval_set[X], eval_set[y])],
            eval_metric='rmse',
            early_stopping_rounds=100,
            verbose=True)

        return model.best_iteration_
    except Exception as e:
        logger.error(f'Exeption in get_n_estimators:{e} ')


def get_params(tenant_id, tune_run_id, cluster_id):
    try:
        model_path = get_root_dir()
        location = '{}/tuning_regression/{}/{}'.format(model_path, tenant_id, tune_run_id)
        with open(location + '/params_{}.pickle'.format(cluster_id), 'rb') as file:
            params = pickle.load(file)

        return params
    except Exception as e:
        logger.info(f'Exeption in get_params: {e} ')


def build_model(config, df_h, X, y, tune_run_id, cluster_id, quantile):
    start_time = time.time()
    try:
        horizon = config.get('horizon')
        tenant_id = config.get('tenant_id')

        train = df_h[X + y]
        print(train.head())
        print('==============getting params')
        params = get_params(tenant_id, tune_run_id, cluster_id)
        print(params)
        print('==============running regression--mse')
        if quantile == 0.5:
            params['objective'] = 'regression'
        else:
            params['objective'] = 'quantile'
            params['metric'] = 'qunatile'
            params['alpha'] = quantile
        print('============n_estimators')
        n_est = get_n_estimators(df_h, X, y, horizon, params)
        print('====================')
        print(n_est)
        params['n_estimators'] = np.max([400, n_est])
        # params['n_estimators'] = 400
        print(train[X].head())
        print(train[y].head())
        print('training_model++++++++++')
        model = lgb.LGBMRegressor(**params)
        model.fit(
            train[X],
            train[y],
            verbose=True)
        print('trained the model--------------------')
        with open("output.txt", "a") as f:
            print(f"build_model completed in {(time.time() - start_time)} seconds", file=f)
        return model
    except Exception as e:
        logger.info(f'Exeption in build_model : {e}' )



def save_model(model, tenant_id, cluster_id, train_run_id, quantile):
    try:
        model_path = get_root_dir()
        location = '{}/training_regression/{}/{}'.format(model_path, tenant_id, train_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(location + '/model_{}_{}.txt'.format(cluster_id, quantile))
    except Exception as e:
        logger.error(f'Exeption in save_model: {e} ' )



def train_for(config, df_h, tune_run_id):
    start_time = time.time()
    try:

        logger.info("training --started")

        tenant_id = config['tenant_id']
        # train_run_id = id_generator()
        train_run_id = tune_run_id
        logger.info("fetch data for training --started")
        cluster_run_id = 1
        X, y = get_Xny(config)
        # df_h, X, y, cluster_run_id = fetch_prepared_data(config, conn, df_input)
        config['cluster_run_id'] = cluster_run_id
        ##########################################
        for cluster_id in df_h.cluster_id.unique():
            try:
                # for quantile in [0.05, 0.5, 0.95]:
                for quantile in [0.5]:
                    df_h_c = df_h.query('cluster_id == @cluster_id')
                    print('==============building model')
                    print('=======df_h_c', df_h_c.head())
                    logger.info("build model --started")
                    model = build_model(config, df_h_c, X, y, tune_run_id, cluster_id, quantile)
                    logger.info("build model --completed")
                    print('==============saving model')
                    save_model(model, tenant_id, cluster_id, train_run_id, quantile)
                    logger.info("save model model --completed")
            except Exception as e:
                logger.info(f'Could not train for cluster={cluster_id}: {e}')
                continue
        with open("output.txt", "a") as f:
            print(f"train_for completed for category:{train_run_id} in {(time.time() - start_time)} seconds", file=f)
        logger.info("training --completed")
        # op_duration_sec = round(time.time() - t0, 3)
        # config_df = create_config_data(config,train_run_id, op_duration_sec, tune_run_id)
        # push_config_table_to_snowflake(config, config_df, conn)
    except Exception as e:
        logger.error(f"Error in train_for:{e}")




