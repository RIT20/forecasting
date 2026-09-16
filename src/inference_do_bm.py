import os
import sys
sys.path.insert(0, os.getcwd())

import traceback
import lightgbm as lgb
from pathlib import Path
import pandas as pd
import time
import numpy as np
import itertools

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger

from utils.io_utils import get_model_dir,get_root_dir
from utils.data_prepare_do_bm import id_generator
from utils.data_prepare_do_bm import push_config_table_to_snowflake, create_config_data


def load_model(tenant_id, train_run_id, cluster_id, quantile):
    try:
        model_path = get_root_dir()
        model = lgb.Booster(model_file='{}/training_do/{}/{}/model_{}_{}.txt'.format(model_path, tenant_id, train_run_id,
                                                                                cluster_id, quantile))
        return model
    except Exception as e :
        logger.error(f'Exeption in load_model:{e} ')

def infer_for(config,df_h, train_run_id):
    try:
        logger.info("Inference --started")
        start_time = time.time()

        tenant_id = config.get('tenant_id')
        # infer_run_id = id_generator()
        # infer_run_id = 'slag1_zerocountvars3'
        infer_run_id = train_run_id

        # df_h = fetch_prepared_data_for_inference(config)
        # import pdb
        # pdb.set_trace()
        config['cluster_run_id'] = 1

        outs = []
        # for cluster_id in df_h.cluster_id.unique():
        for cluster_id in range(1,2):
            try:
                # for quantile in [0.05, 0.5, 0.95]:
                for quantile in [0.5]:
                    cluster_id = int(cluster_id)
                    # df_h_c = df_h.query('cluster_id == @cluster_id')
                    df_h_c = df_h
                    model = load_model(tenant_id, train_run_id, cluster_id, quantile)
                    logger.info("load model --completed")
                    X = model.feature_name()

                    pred_results = df_h_c.assign(
                        pred_sales = model.predict(df_h_c[X])
                    )
                    out = pred_results[['product_id', 'effective_date', 'pred_sales']]
                    out['quantile'] = quantile
                    outs.append(out)
            except Exception as e:
                print('Could not train for cluster: {}'.format(cluster_id), e)
                continue

        out = pd.concat(outs)

        ##Denormalize the sales
        # out = denormalize(out, config, conn)

        out = out.assign(
            tenant_id = tenant_id,
            # infer_run_id = infer_run_id,
            # train_run_id = train_run_id,
            # cluster_run_id = config['cluster_run_id'],
            # forecast_type = 'weekly-forecast',
            duration = 7,
            created_at = pd.Timestamp.now()
        ).rename(
            columns = {
                'product_id': 'item',
                'pred_sales': 'quantity'
            }
        )


        cols_order = ['item','tenant_id',
                      # 'forecast_type',
                      'duration',
                      'quantity',
                      'effective_date',
                    # 'infer_run_id','train_run_id',
                    # 'cluster_run_id',
                      'created_at', 'quantile']
        out = out[cols_order]

        # out.to_csv(f'forecasts/do/{infer_run_id}.csv')
        root_path = get_root_dir()
        folder_name = tenant_id+ '_' + str(config.get('infer_start_date'))
        location = '{}/forecast_do/{}/'.format(root_path, folder_name)
        Path(location).mkdir(parents=True, exist_ok=True)
        file = str(location + f'{infer_run_id}_forecast.csv')
        out.to_csv(file, index=False)
        with open("output_do.txt", "a") as f:
            print(f"infer_for completed in {(time.time() - start_time)} seconds", file=f)
        logger.info("Inference --completed")
        # out = disaggregate(out, config, conn)
        # out = out[cols_order]

        # out1, out2 = get_outputs(out, config)
        # save_data_and_push_to_snowflake(config, out1, out2, infer_run_id, conn)

        # op_duration_sec = round(time.time() - t0,3)
        # config_df = create_config_data(config, infer_run_id, op_duration_sec, train_run_id)
        # push_config_table_to_snowflake(config,config_df, conn)
    except Exception as e :
        logger.error(f"Error in infer_for: {e}")


