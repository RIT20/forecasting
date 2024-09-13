from utils import data_prepare_regression_bm as dp
from memory_profiler import profile
from core_utils.config_manager import DBManager
from src import tuning_regression_bm
from src import training_regression_bm
from src import inference_regression_bm
import pandas as pd
import tracemalloc
from core_utils.logger import logger
import time
import os

tracemalloc.start()
tenant_id = 'GLOBAL_PARTNER'
start_date = str(pd.Timestamp(2018, 1, 1).date())
forecast_for_last_week = str(pd.Timestamp(2023, 2, 26))
default_config = {
    'db_name': 'dev_global_partner',
    'category_col': 'UDA_DEPT_DESC',
    'start_date': start_date,
    'tenant_id': tenant_id,
    'horizon': 1,
    'tune_method': 'optuna',
    'sales_lags': 10,
    'loc_count_lags': 30,
    'futures': 7,
    'tuning_n_trials': 10,
    'cluster_run_id': 1,
    'env_name': 'DEV',
    'prepared_data_table_name': 'temp_prepared_data_regression',
    'data_save_method': 'local',
    'infer_start_date': forecast_for_last_week
}
def prepare_config(config,tenant_id,env,infer_start_date,horizon,tenant_type="B&M"):
    config["tenant_id"] = tenant_id
    logger.info(f'------ Regression Job Started for tenant={config["tenant_id"]} at {pd.Timestamp("today").strftime("%Y-%m-%d %H:%M:%S.%02f")[:-4]}-----')
    with open("output.txt", "a") as f:
        print(f'------Job Started at {pd.Timestamp("today").strftime("%Y-%m-%d %H:%M:%S.%02f")[:-4]}-----', file=f)
    DBManager.set_customer(customer=tenant_id, env=env)
    config["env_name"] = env
    if tenant_type=="B&M":
        config["db_name"] = env +"_" + tenant_id
    else:
        config["db_name"] = env + "_" + "ECOM"
    config["horizon"] = horizon
    config["infer_start_date"] = infer_start_date
    config['end_date'] = str(pd.Timestamp(config.get('infer_start_date')) - pd.Timedelta(days=1 * 7))

    conn = getattr(DBManager, "engine")
    config['conn'] = conn
    return config

def tune_train_infer(config, category):
    logger.info(f'################### Tuning,training,inference started For Category: {category}')
    start_time = time.time()
    conn = getattr(DBManager, "engine")
    formatted_category = dp.get_formatted_str(category)
    train_data, infer_data = dp.fetch_prepared_data_from_source(config,category,formatted_category,conn,config["data_save_method"])

    ############tuning--
    config['infer_start_date'] = str(config.get('infer_start_date'))
    config['end_date'] = str((pd.Timestamp(config.get('infer_start_date')) - pd.Timedelta(days=7)).date())
    config['op_name'] = 'tuning'
    tune_run_id = formatted_category
    tuning_regression_bm.tune_for(config, train_data, tune_run_id)
    with open("output.txt", "a") as f:
        print(f' tuning memory_info :{tracemalloc.get_traced_memory()}', file=f)

    ##############training--
    config['op_name'] = 'training'
    print('Training Started')
    training_regression_bm.train_for(config, train_data, tune_run_id)
    print('Training Ended')

    with open("output.txt", "a") as f:
        print(f' training memory_info :{tracemalloc.get_traced_memory()}', file=f)

    #########inference--
    config['infer_start_date'] = str(config.get('infer_start_date'))
    config['op_name'] = 'inference'
    print('Inference Started')
    inference_regression_bm.infer_for(config,infer_data, tune_run_id)
    print('Inference Ended')
    logger.info(f'################### Tuning,training,inference completed For Category: {category}')
    with open("output.txt", "a") as f:
        print(f' inference memory_info :{tracemalloc.get_traced_memory()}', file=f)
        print(f"category {formatted_category} -- tune-train-inference completed in {(time.time() - start_time) / 60} minutes", file=f)

def create_chunk_list(a, chunk_size):
    if chunk_size < 2 or chunk_size % 2 != 0:
        raise ValueError("invalid chunk size")
        return
    chunks = []
    range_size = int(chunk_size / 2)
    while len(a) > 0:
        if len(a) <= chunk_size:
            chunks.append(a)
            return chunks
        else:
            chunk_start = [a.pop(0) for i in range(range_size)]
            chunk_end = [a.pop(-1) for i in range(range_size)]
            chunks.append(chunk_start + chunk_end)

    return chunks

@profile
def driver_new(config):
    start_time = time.time()
    conn = config["conn"]
    cat_pairs_full = dp.fetch_store_category_pairs(config, conn)
    print(cat_pairs_full)
    dp.drop_intermediate_table_before_run(config, conn)
    cat_pairs_full_list = cat_pairs_full["category"].to_list()
    # cat_pairs_full_list = cat_pairs_full_list[:10]
    # cat_pairs_full_list = ['NATURAL  ORGANIC DAIRY']

    cat_pairs_full_list_copy = cat_pairs_full_list[:]
    print(cat_pairs_full_list)
    category_chunks = create_chunk_list(cat_pairs_full_list_copy,2)
    print(category_chunks)
    try:
        os.remove(f"no_data_cat_{config['tenant_id']}.txt")
    except Exception as e:
        logger.error(f"Error while deleting the exclude cat file : {e}")

    for cat_chunk in category_chunks:
        dp.prepare_initial_data(config, conn, cat_chunk)

    exclude_cat_list = []
    try:
        with open(f"no_data_cat_{config['tenant_id']}.txt", "r") as f:
            for line in f:
                x = line[:-1]
                exclude_cat_list.append(x)
    except Exception as e:
        logger.error(f"Error while reading the exclude cat file : {e}")

    for category in cat_pairs_full_list:
        # category = row['category']
        # category = '24 Automotive Products'
        # category = '28 Ice(Purch/Bagged From Vndr)'
        if category in exclude_cat_list:
            continue
        with open("output.txt", "a") as f:
            print(f'################### Tuning,training,inference For Category: {category}', file=f)
        tune_train_infer(config, category)
    with open("output.txt", "a") as f:
        print(f"Regression for tenant_id : {config['tenant_id']} completed in {(time.time() - start_time) / 60} minutes", file=f)


forecast_start_date = str(pd.Timestamp(2023, 2, 26))
config = prepare_config(config=default_config,tenant_id="WOODMAN",env= "DEV", horizon=4, infer_start_date= forecast_start_date)
# import pdb;pdb.set_trace()
driver_new(config)

tracemalloc.stop()