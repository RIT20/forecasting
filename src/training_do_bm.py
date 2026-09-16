import os
import sys
sys.path.insert(0, os.getcwd())
import traceback

import logging
import logging.config 
from pathlib import Path
import pickle

import lightgbm as lgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import time

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger

from utils.data_prepare_do_bm import id_generator, get_Xny
from utils.data_prepare_do_bm import push_config_table_to_snowflake, create_config_data, convert_cat_features_dtype
from imblearn.over_sampling import SMOTENC
from evalml.pipelines.components.transformers.samplers import Oversampler
from utils.io_utils import get_model_dir,get_root_dir

def get_smoted_samples_memory_effecient_impl(config, df):
    try:
        start_time = time.time()
        cat_features = ['lag1', 'is_covid',
                        'is_holiday_fut_0', 'is_holiday_fut_1', 'is_holiday_fut_2',
                        'is_holiday_fut_3', 'is_holiday_fut_4', 'is_holiday_fut_5',
                        'is_holiday_fut_6', 'product_id']
        for cat in cat_features:
            df[cat] = df[cat].astype('category')
        X, y = get_Xny(config)
        N=10
        list_of_new_samples = []
        ones = df[df.sales==1]
        zeros = df[df.sales==0]
        for i in range(N+1):
            sample_ones = ones.sample(frac=1/N, replace=True, random_state=42)
            sample_zeros = zeros.sample(frac=1/N, replace=True, random_state=42)
            sample = pd.concat([sample_ones, sample_zeros])
            oversample_df, y_new = Oversampler(sampling_ratio=0.5,n_jobs=35).fit_transform(
                sample[X],
                sample[y].values.ravel(),
                )
            oversample_df['sales'] = y_new
            oversample_df = convert_cat_features_dtype('', oversample_df)
            full_df = oversample_df.merge(sample.drop_duplicates(), on=X+y, how='left', indicator=True)
            smote_generated_samples = full_df[full_df['_merge'] == 'left_only']
            list_of_new_samples.append(smote_generated_samples)

        smote_generated_samples = pd.concat(list_of_new_samples)
        # df = df[X+y]
        df_train = pd.concat([smote_generated_samples, df])
        del df
        cat_features.pop(-1)
        for cat in cat_features:
            df_train[cat] = df_train[cat].astype(int)
        df_train = convert_cat_features_dtype('', df_train)
        df_train = df_train[X+y]
        with open("output_do.txt", "a") as f:
            print(f"get_smoted_samples completed in {(time.time() - start_time)} seconds", file=f)
        return df_train
    except Exception as e:
        logger.error(f"Exception in get smote samples: {e}")
def get_smoted_samples(config, df):
    try:
        start_time = time.time()
        print('smoted start')
        X, y = get_Xny(config)

        a = df.sales.value_counts().values
        if a[1] / a[0] > 0.5:
            return df

        cat_features = ['lag1', 'is_covid',
            'is_holiday_fut_0', 'is_holiday_fut_1', 'is_holiday_fut_2',
            'is_holiday_fut_3', 'is_holiday_fut_4', 'is_holiday_fut_5',
            'is_holiday_fut_6', 'product_id']

        print('smoted middle')
        df_train = df
        cat_features_mask = [True if feat in cat_features else False for feat in df_train[X].columns]
        train_df, y_new = SMOTENC(categorical_features = cat_features_mask, sampling_strategy=0.5).fit_resample(
            df_train[X],
            df_train[y].values.ravel(),
            )

        train_df['sales'] = y_new
        df_train = train_df
        df_train = convert_cat_features_dtype('', df_train)
        with open("output_do.txt", "a") as f:
            print(f"get_smoted_samples completed in {(time.time() - start_time)} seconds", file=f)
        return df_train
    except Exception as e:
        logger.error(f"Exception in get smote samples: {e}")

def get_n_estimators(config, df, X, y, horizon, params):
    try:
        start_time = time.time()
        print('get_n_est called4')
        horizon = 4                    
        eval_cutoff = sorted(df['effective_date'].unique())[-horizon:][0]
        train = df.query('effective_date < @eval_cutoff')
        # train = get_smoted_samples(config, train)
        # train = get_smoted_samples_memory_effecient_impl(config, train)
        eval_set = df.query('effective_date >= @eval_cutoff')
        # eval_cutoff_idx = int((95/100)*len(df))
        # print(eval_cutoff_idx)
        # train = df.iloc[:eval_cutoff_idx]
        # eval_set = df.iloc[eval_cutoff_idx:]
        
        # model = lgb.LGBMRegressor(**params)
        # model.fit(
        #         train[X], 
        #         train[y], 
        #         eval_set=[(c[X], eval_set[y])], 
        #         eval_metric='rmse',                
        #         early_stopping_rounds=30,
        #         verbose=True)

        
        model = lgb.LGBMClassifier(**params)
        model.fit(
                train[X], 
                train[y],
                eval_set=[(eval_set[X], eval_set[y])],
                early_stopping_rounds=30
            )
        with open("output_do.txt", "a") as f:
            print(f"get_n_estimators completed in {(time.time() - start_time)} seconds", file=f)
        return model.best_iteration_
    except Exception as e :
        logger.error(f'Exeption in get_n_estimators:{e} ')

def get_params( tenant_id, tune_run_id, cluster_id):
    try:
        model_path = get_root_dir()
        location = '{}/tuning_do/{}/{}'.format(model_path,tenant_id, tune_run_id)
        with open(location + '/params_{}.pickle'.format(cluster_id), 'rb') as file:
            params = pickle.load(file)

        return params
    except Exception as e :
        logger.error(f'Exeption in get_params:{e} ')

def build_model(config, df_h, X, y, tune_run_id, cluster_id, quantile):
    try:
        start_time = time.time()
        horizon = config.get('horizon')
        tenant_id = config.get('tenant_id')


        # train = df_h[X + y]
        params = get_params( tenant_id, tune_run_id, cluster_id)
        # params = {
        #     'n_estimators': 3000,
        #     'learning_rate': 0.03,
        #     'num_leaves': 60,
        #     'max_depth': 7,
        #     'min_data_in_leaf': 1,
        #     'bagging_fraction': 0.7,
        #     'bagging_freq': 1
        # }

        # if quantile == 0.5:
        #     params['objective'] = 'regression'
        # else:
        #     params['objective'] = 'quantile'
        #     params['metric'] = 'qunatile'
        #     params['alpha'] = quantile

        n_est = get_n_estimators(config, df_h,X, y, horizon, params )
        params['n_estimators'] = np.max([400, n_est])
        # params['n_estimators'] = n_est

        # model = lgb.LGBMRegressor(**params)
        # df_h = get_smoted_samples(config, df_h)
        # df_h = get_smoted_samples_memory_effecient_impl(config,df_h)
        model = lgb.LGBMClassifier(**params)
        model.fit(
                df_h[X], 
                df_h[y], 
                verbose=True)
        with open("output_do.txt", "a") as f:
            print(f"build_model completed in {(time.time() - start_time)} seconds", file=f)
        return model
    except Exception as e :
        logger.error(f'Exeption in build_model:{e} ')


def save_model(model, tenant_id, cluster_id, train_run_id, quantile):
    try:
        model_path = get_root_dir()
        location = '{}/training_do/{}/{}'.format(model_path, tenant_id, train_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(location + '/model_{}_{}.txt'.format(cluster_id, quantile))
    except Exception as e :
        logger.error(f'Exeption in save_model:{e} ')


def train_for(config,tune_run_id, df_h):
    try:
        logger.info("Training --started")
        start_time = time.time()

        tenant_id = config['tenant_id']
        train_run_id = tune_run_id

        X,y = get_Xny(config)
        cluster_run_id = 1
        # df_h, X, y, cluster_run_id = fetch_prepared_data(config, conn, df)
        config['cluster_run_id'] = cluster_run_id

        # for cluster_id in df_h.cluster_id.unique():
        for cluster_id in range(1,2): # why this step??
            try:
                # for quantile in [0.05, 0.5, 0.95]:
                for quantile in [0.5]:
                    # df_h_c = df_h.query('cluster_id == @cluster_id')
                    df_h_c = df_h
                    logger.info("build model --started")
                    model = build_model(config, df_h_c, X, y, tune_run_id, cluster_id, quantile)
                    logger.info("build model --completed")
                    save_model(model, tenant_id, cluster_id, train_run_id, quantile)
                    logger.info("save model --completed")

            except Exception as e:
                logger.info('Could not train for cluster: {}'.format(cluster_id), e)
                continue
        with open("output_do.txt", "a") as f:
            print(f"train_for completed in {(time.time() - start_time)} seconds", file=f)
        logger.info("Training --completed")
        # config_df = create_config_data(config,train_run_id, op_duration_sec, tune_run_id)
        # push_config_table_to_snowflake(config, config_df, conn)
    except Exception as e:
        logger.error(f"Error in train_for: {e}",)
