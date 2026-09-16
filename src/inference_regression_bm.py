import os
import sys

import traceback
import lightgbm as lgb
from pathlib import Path
import pandas as pd
import time
import numpy as np
import itertools

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger

from utils.data_prepare_regression_bm import id_generator
from utils.data_prepare_regression_bm import push_config_table_to_snowflake, create_config_data
from utils.etl import get_etl_Pipeline
from utils.io_utils import get_model_dir, get_root_dir


def load_model(tenant_id, train_run_id, cluster_id, quantile):
    start_time = time.time()
    try:
        model_path = get_root_dir()
        model = lgb.Booster(model_file='{}/training_regression/{}/{}/model_{}_{}.txt'.format(model_path, tenant_id, train_run_id,
                                                                                 cluster_id, quantile))
        return model
    except Exception as e:
        logger.info(f'Exeption in load_model:{e} ')
    print(f"load_model completed in {(time.time() - start_time)} seconds")


def infer_for(config, df_h, train_run_id):
    start_time = time.time()
    try:

        logger.info("inference --started")
        # if 'conn' in config.keys():
        #     conn = config['conn']
        # else:
        #     conn = get_snowflake_connection_string(config.get('env_name'))
        tenant_id = config.get('tenant_id')
        infer_run_id = train_run_id
        cluster_run_id =1
        conn = getattr(DBManager, "engine")
        logger.info("fetch data for inference --started")
        # df_h, cluster_run_id = fetch_prepared_data_for_inference2(config, conn)
        logger.info("fetch data for inference --completed")
        # import pdb
        # pdb.set_trace()
        ####################################################
        # df_h = df_h
        # cluster_run_id = 1
        #####################################################

        # df_h, cluster_run_id = fetch_prepared_data_for_inference(config)
        config['cluster_run_id'] = cluster_run_id

        # train_run_id = get_train_runid_from_snowflake(config, conn)
#         train_run_id = config.get('tenant_id') + config.get('infer_start_date')
        # train_run_id = config['train_run_id']
        # train_run_id = 'UBU3DAFPML'

        outs = []
        for cluster_id in df_h.cluster_id.unique():
            try:
                # for quantile in [0.05, 0.5, 0.95]:
                for quantile in [0.5]:
                    cluster_id = int(cluster_id)
                    df_h_c = df_h.query('cluster_id == @cluster_id')
                    print('before model######')
                    logger.info("load model --started")
                    model = load_model(tenant_id, train_run_id, cluster_id, quantile)
                    logger.info("load model --completed")
                    print('after model######')
                    X = model.feature_name()
                    pred_results = df_h_c.assign(
                        pred_sales=model.predict(df_h_c[X])
                    )
                    out = pred_results[['product_id', 'effective_date', 'pred_sales']]
#                     out['quantile'] = quantile
                    outs.append(out)
            except Exception as e:
                print(f'Could not train for cluster={cluster_id}: {e}')
                continue

        out = pd.concat(outs)
        out = out.assign(
            tenant_id=tenant_id,
#             infer_run_id=infer_run_id,
#             train_run_id=train_run_id,
#             cluster_run_id=config['cluster_run_id'],
#             forecast_type='weekly-forecast',
            duration=7,
            created_at=pd.Timestamp.now()
        ).rename(
            columns={
                'product_id': 'item',
                'pred_sales': 'quantity'
            }
        )

        cols_order = ['item',
                      'tenant_id',
#                       'forecast_type',
                      'duration',
                      'quantity',
                      'effective_date',
#                       'infer_run_id',
#                       'train_run_id',
#                       'cluster_run_id',
                      'created_at',
#                       'quantile'
                     ]
        out = out[cols_order]
        tenant_id = out['tenant_id'].unique()[0]
        maxed_date = str(out['effective_date'].max())

        root_path = get_root_dir()
        folder_name = tenant_id + '_' + str(config.get('infer_start_date'))
        location = '{}/forecast_regression/{}/'.format(root_path, folder_name)
        Path(location).mkdir(parents=True, exist_ok=True)
        file = str(location+f'{infer_run_id}_forecast.csv')
        out.to_csv(file, index=False)
        with open("output.txt", "a") as f:
            print(f"infer_for completed in {(time.time() - start_time)} seconds", file=f)
#         out.to_csv(f'/home/nitesht/regression/yearly_forecast/{tenant_id}_{maxed_date}.csv')
        logger.info("inference --completed")
        # out = disaggregate(out, config, conn)
        # out = out[cols_order]
        # print('output after disaggregation#####')
        # out1, out2 = get_outputs(out, config)
        # # out1.to_csv('')
        # # out2.to_csv('')
        # print('out1#####')
        # print(out1.head())
        # save_data_and_push_to_snowflake(config, out1, out2, infer_run_id, conn)

        # op_duration_sec = round(time.time() - t0,3)
        # config_df = create_config_data(config, infer_run_id, op_duration_sec, train_run_id)
        # push_config_table_to_snowflake(config,config_df, conn)
    except Exception as e:
        print("Error in infer_for: %s", traceback.format_exc())
        logger.error(f"Error in infer_for:{e}")

def disaggregate(config, df, category):
    conn = config.get('conn')
    tenant_id = config.get('tenant_id')
    # config = configuration.model.to_dict()
    params = {
        "tenant_id": tenant_id,
        "infer_start_sunday": config.get("infer_start_date"),
        "disagg_consideration_window": config.get("sales_lags"),
        "category":str(category),
        'db_name':config.get('db_name')
    }
    qry = ''' 
    with t1 as (
    select
        item || '$#$' || loc || '$#$' || channel as item,
        MOD(DAYOFWEEK(effective_date)+1, 7) as dayofweek,
        quantity
  
    from
        (
            SELECT
                *
            FROM
                {db_name}.ml.fct_sales
            where
                (item, loc, channel) in (
                    select
                        distinct item,
                        loc,
                        channel
                    from
                        {db_name}.ml.dim_work_unit
                    where
                        status = 1
                        and tenant_id = '{tenant_id}'
                        and UDA_DEPT_DESC ='{category}'
                )
        )
    where
        item is not null
        and item != ''
        and effective_date >= dateadd(week,-'{disagg_consideration_window}', to_date('{infer_start_sunday}')) and  effective_date < to_date('{infer_start_sunday}')
        and tenant_id = '{tenant_id}'
        and net_sales >= 0
        and item != 'SALES TAX'
    )
    select         
        item,
        dayofweek,
        avg(quantity) as avg_sales,
        sum(quantity) as total_sales,
        count(dayofweek) as count_days_data_avl
    from t1
    group by item, dayofweek
    '''.format(**params)
    
    df_w1 = pd.read_sql(qry, conn)

    prods = df_w1.item.unique()
    week_nums = [0, 1, 2, 3, 4, 5, 6]

    df_w2 = pd.DataFrame(
        itertools.product(prods, week_nums), columns=["item", "dayofweek"]
    )
    df_w2 = df_w2.merge(df_w1, how="left", on=["item", "dayofweek"])[
        ["item", "dayofweek", "total_sales"]
    ]
    df_w2["total_sales"] = df_w2["total_sales"].fillna(0)
    df_w2["total_sales_in_window"] = df_w2.groupby("item").total_sales.transform("sum")
    df_w2["weight"] = df_w2["total_sales"] / df_w2["total_sales_in_window"]
    df_w2["weight"] = df_w2.weight.replace([np.inf, np.nan], 1 / 7)
    df_w2 = df_w2.drop(columns=["total_sales", "total_sales_in_window"])

    start_dt = df.effective_date.min()
    end_dt = df.effective_date.max() + pd.Timedelta(days=6)

    prods = df.item.unique()
    dates = pd.date_range(start=start_dt, end=end_dt, freq="D")

    df_w3 = pd.DataFrame(
        itertools.product(prods, dates), columns=["item", "effective_date"]
    )
    df_w3["dayofweek"] = np.mod(df_w3.effective_date.dt.dayofweek + 1, 7)
    df_w3 = df_w3.merge(df_w2, how="left", on=["item", "dayofweek"])
    df_w3["week_sunday"] = df_w3["effective_date"].apply(
        lambda x: pd.Timestamp(
            (x.date() - pd.Timedelta(days=np.mod(x.dayofweek + 1, 7)))
        )
    )

    df = df.rename(columns={"effective_date": "effective_date_old"})
    df = df.merge(
        df_w3,
        how="left",
        left_on=["item", "effective_date_old"],
        right_on=["item", "week_sunday"],
    )
    ##########
    df['weight'] = df.weight.replace([np.inf, np.nan], 1/7)
    ###########
    df["forecast_type"] = getattr(configuration.model, "forecast_type")
    df["duration"] = 1
    df["quantity_old"] = df["quantity"]
    df["quantity"] = df["quantity"] * df["weight"]
    df = df.drop(columns=["effective_date_old", "week_sunday", "weight", "dayofweek"])

    return df



def merge_reg_do(config,tune_run_id, category):

    tenant_id = config.get('tenant_id')
    root_path = get_root_dir()
    folder_name = tenant_id+ '_' + str(config.get('infer_start_date'))
    infer_run_id = tune_run_id
    
    # do
    location = '{}/forecast_do/{}/'.format(root_path, folder_name)
    file_do = str(location + f'{infer_run_id}_forecast.csv')
    df_do = pd.read_csv(file_do, parse_dates=['effective_date'])
    
    # regression
    folder_name = tenant_id + '_' + str(config.get('infer_start_date'))
    location = '{}/forecast_regression/{}/'.format(root_path, folder_name)
    file = str(location+f'{infer_run_id}_forecast.csv')
    df = pd.read_csv(file, parse_dates=['effective_date'])

    # merging and disaggregating
    df_do = df_do.rename(columns = {'quantity':'proba_do'})[['item', 'effective_date', 'proba_do']]
    df = df.rename(columns = {'quantity':'pred_sales'})
    df = df.merge(df_do, how='left', on=['item', 'effective_date'])
    df['proba_do'] = df['proba_do'].fillna(1)
    df['pred_sales'] = df['pred_sales']*df['proba_do']
    df.to_csv(location + '/test.csv')
    df = df.drop(columns=['proba_do'])
    df['quantity'] = df['pred_sales']
    out = disaggregate(config, df, category)
    out['pred_sales'].mask(out['pred_sales']<0, 0, inplace=True)
    
    # saving csv in the folder of forecast_regression
    Path(location).mkdir(parents=True, exist_ok=True)
    file = str(location+f'{infer_run_id}_doforecast.csv')
    out.to_csv(file, index=False)

    logger.info("inference --completed")
    return out