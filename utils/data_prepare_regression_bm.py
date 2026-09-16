import os
import sys

sys.path.insert(0, os.getcwd())

import time
from memory_profiler import profile
import pathlib
from pathlib import Path
import string
import pandas as pd
import numpy as np
import holidays
import itertools
import datetime

from utils.io_utils import get_sql_dir,get_root_dir
from utils.db import write_to_snowflake,execute_sql_on_snowflake
from utils.hpc import parallelize
from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger

category_sql = pathlib.Path(os.path.join(get_sql_dir(), 'category_query.sql')).read_text(encoding='utf-8')

category_raw_data_sql = pathlib.Path(os.path.join(get_sql_dir(), 'feature_query_cat_regression.sql')).read_text(encoding='utf-8')


def get_formatted_str(s):
    s = s.replace('/', '_')
    s = s.replace("\\", "_")
    s = s.replace(' ', '_')
    return s


def fetch_store_category_pairs(config, conn):
    logger.info('fetching distinct categories...')
    #     df = pd.read_csv('store_cat.csv')
    qry = category_sql.format(
        tenant_id=config.get('tenant_id'),
        db_name=config.get('db_name'),
        category_col=config.get('category_col'))
    print(qry)
    df = pd.read_sql(qry, conn)
    logger.info('Completed - fetching distinct categories...')
    return df


def features_level_info(config, level):
    features = {
        'prod_level': [
            'cluster_id',
            'first_sale_date',
            # 'weight', 'weight_uom',
            # 'uda_level1', 'uda_level2', 'uda_level3', 'uda_level4', 'uda_level5','uda_level6',
            # 'uda_attribute1', 'uda_attribute2', 'uda_attribute3','uda_attribute4', 'uda_attribute5'
        ],
        'date_level': ['is_holiday_fut_{}'.format(i) for i in range(config.get('futures'))],
        'prod_date_level': [
            'sales',
        ],
        'cat_features': [
            'product_id',
            # 'covid_var'
            # 'weight_uom',
            # 'uda_level1', 'uda_level2', 'uda_level3', 'uda_level4', 'uda_level5','uda_level6',
            # 'uda_attribute1', 'uda_attribute2', 'uda_attribute3','uda_attribute4', 'uda_attribute5'
        ]
    }

    return features[level]




def out_treat(df_copy):
    try:
        df_copy["c1"] = df_copy.groupby(["product_id"])["sales"].transform('mean')
        df_copy["c2"] = df_copy.groupby(["product_id"])["sales"].transform('std')
        df_copy["c1"] = (df_copy["sales"] - df_copy["c1"]) / df_copy["c2"]
        df_copy["c2"] = np.where(df_copy['c1'] > 2, True, False)
        df_copy["c1"] = df_copy[df_copy["c2"] == False].groupby(["product_id"])["sales"].transform('mean')
        df_copy["c1"] = df_copy.groupby(["product_id"])["c1"].apply(lambda x: x.ffill().bfill())
        df_copy["sales"] = (df_copy["c2"] * df_copy['c1']) + ((~df_copy["c2"]) * df_copy["sales"])
        df_copy = df_copy.drop(["c1", "c2"], axis=1)

        return df_copy
    except Exception as e:
        logger.error(f"error during outlier treatment: {e}")


def fetch_cat_raw_data(config, conn):
    try:
        start_time = time.time()
        if config['start_date'] == None:
            config['start_date'] = pd.Timestamp(config.get('end_date')) - pd.Timedelta(days=365 * 5)

        if config['cluster_run_id'] == None:
            # config['cluster_run_id'] = get_clusterid_from_snowflake(config, con=conn)
            config['cluster_run_id'] = 1

        end_date = pd.Timestamp(config.get('end_date')) + pd.Timedelta(days=6)
        active_products_start_date = pd.Timestamp(end_date) - pd.Timedelta(days=52 * 3 * 7)
        # end_date = pd.Timestamp(config.get('end_date'))
        start_time = time.time()
        conn.execute(''' ALTER SESSION SET WEEK_START=7   ''')

        def construct_query(select_col, limit_query):
            qry = category_raw_data_sql.format(
                db_name=config.get('db_name'),
                start_date=config.get('start_date'),
                end_date=end_date,
                tenant_id=config.get('tenant_id'),
                cluster_run_id=config.get('cluster_run_id'),
                category_col=config.get('category_col'),
                category = config.get('category'),
                active_products_start_date=active_products_start_date,
                select_col=select_col,
                limit_query=limit_query
            )
            return qry

        count_select = 'count(*) as row_count'
        count_qry = construct_query(count_select, '')
        print(count_qry)
        row_count = pd.read_sql(count_qry, conn)["row_count"].unique()[0]

        def create_chunk_offset_list(row_count, num_parts):
            part_size = row_count // num_parts  # calculate the size of each part
            remainder = row_count % num_parts
            parts = [part_size] * num_parts
            for i in range(remainder):
                parts[i] += 1
            initial_offset_list = [0] + parts[:-1]
            total = 0
            offset_list = []
            for i in initial_offset_list:
                total = total + i
                offset_list.append(total)
            return parts, offset_list

        def fetch_data_in_parts(offset, fetch_count):
            conn.execute(''' ALTER SESSION SET WEEK_START=7   ''')
            select_qry = "product_id,week_effective_date,sales"
            limit_qry = f"offset {offset} rows fetch next {fetch_count} rows only"
            data_qry = construct_query(select_qry, limit_qry)
            print(data_qry)
            df = pd.read_sql(data_qry, conn)
            return df

        threshold = 2000000
        no_of_chunks = 32
        if row_count > threshold:
            chunk_list, offset_list = create_chunk_offset_list(row_count, no_of_chunks)

            params = []
            for offset, fetch_count in zip(offset_list, chunk_list):
                params.append({"offset": offset, "fetch_count": fetch_count})
            raw_data_list = parallelize(fetch_data_in_parts, params=params, backend="threading", n_jobs=4)
            df = pd.concat(raw_data_list)
        else:
            df = fetch_data_in_parts(0, row_count)

        df = df.reset_index(drop=True)

        logger.info("data fetch --completed")
        with open("output.txt", "a") as f:
            print(f"data fetch completed for cateogry:{config['category']} in {(time.time() - start_time)} seconds", file=f)
        with open("output.txt", "a") as f:
            print(f"no of timeseries in raw_data for cateogry:{config['category']} =  {len(df['product_id'].unique())} ", file=f)

        # Remove items with id null or empty string

        df = df[pd.notnull(df.product_id) & (df.product_id != '')]

        df['week_effective_date'] = pd.to_datetime(df['week_effective_date'])

        df['first_sale_date'] = df.groupby('product_id').week_effective_date.transform('min')
        df['first_sale_date'] = pd.to_datetime(df['first_sale_date'])

        df = df.assign(cluster_id=1)
        df['cluster_id'] = df['cluster_id'].astype('int')
        # import pdb;pdb.set_trace()
        df["sales"] = df["sales"].fillna(0)
        df = df.rename(columns={'week_effective_date': 'effective_date'})
        logger.info(f"fetch_cat_raw_data for cateogry:{config['category']} --completed")
        with open("output.txt", "a") as f:
            print(f"fetch cat_raw_data completed for cateogry:{config['category']} in {(time.time() -start_time) } seconds", file=f)
        return df
    except Exception as e:
        logger.error(f'Exeption in fetch_cat_raw_data: {e} ')


def add_holiday_features(config, df):
    start_time = time.time()
    try:
        us_holidays = holidays.US()

        def get_week_holiday_count(week_sunday):
            cnt = 0
            for i in range(7):
                dt = week_sunday + pd.Timedelta(days=i)
                if dt in us_holidays:
                    cnt = cnt + 1
            return cnt

        futures = config.get('futures')
        all_dates = df.effective_date.unique()
        holiday_list = []
        for dt in all_dates:
            all_holidays = {}
            all_holidays['effective_date'] = dt

            for future in range(futures):
                dt_new = dt + pd.Timedelta(days=7 * future)
                holiday_count = get_week_holiday_count(dt_new)
                all_holidays['is_holiday_fut_{}'.format(future)] = holiday_count
            holiday_list.append(all_holidays)
        all_holidays = pd.DataFrame(holiday_list)

        df = df.merge(all_holidays, how='left', on='effective_date')

        with open("output.txt", "a") as f:
            print(f"add_holiday_features completed in {(time.time() - start_time)} seconds", file=f)
        return df
    except Exception as e:
        logger.error(f'Exeption in add_holiday_features:{e} ')


def get_prod_level_features(config, df):
    start_time = time.time()
    try:
        prod_level_list = features_level_info(config, 'prod_level')
        df_prod_level = df.drop_duplicates('product_id')[['product_id'] + prod_level_list]
        with open("output.txt", "a") as f:
            print(f"get_prod_level_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_prod_level
    except Exception as e:
        logger.error(f'Exeption in get_prod_level_features:{e}')


def get_week_level_features(config, df):
    start_time = time.time()
    try:
        date_level_list = features_level_info(config, 'date_level')
        df_week_level = df.drop_duplicates('effective_date')[['effective_date'] + date_level_list]
        with open("output.txt", "a") as f:
            print(f"get_week_level_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_week_level
    except Exception as e:
        logger.error(f'Exeption in get_week_level_features:{e} ')


def get_prod_week_level_features(config, df):
    start_time = time.time()
    try:
        prod_date_level_list = features_level_info(config, 'prod_date_level')
        df_prod_week_level = df[['product_id', 'effective_date'] + prod_date_level_list]
        with open("output.txt", "a") as f:
            print(f"get_prod_week_level_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_prod_week_level
    except Exception as e:
        logger.error(f'Exeption in get_prod_week_level_features:{e} ')


def handle_missing_values(config, df):
    start_time = time.time()
    try:
        horizon = config.get("horizon")
        ################## find the ends of each time series #############################
        df = df.assign(
            ts_start=df.groupby('product_id').effective_date.transform('min'),
            # ts_end = df.groupby('product_id').effective_date.transform('max'),
            ts_end=pd.Timestamp(config["infer_start_date"]) + pd.Timedelta(days=(horizon-1) * 7),

        )

        df_products_copy = df[['product_id', 'ts_start', 'ts_end']].drop_duplicates()
        df_products_copy["effective_date"] = df_products_copy.apply(
            lambda x: pd.date_range(x.ts_start, x.ts_end, freq='W-SUN'), axis=1)
        df_products_copy = df_products_copy.drop(['ts_start', 'ts_end'], axis=1)
        df_products_copy = df_products_copy.set_index(['product_id']).apply(pd.Series.explode).reset_index()

        df = df_products_copy.merge(df, how='left', on=['product_id', 'effective_date'])
        del df_products_copy

        #######################Impute the sales ##############################################
        prod_date_level_list = features_level_info(config, 'prod_date_level')
        for feat in prod_date_level_list:
            # df[feat] = df.sort_values(['product_id', 'effective_date'])[feat].fillna(method='ffill')
            df[feat] = df[feat].fillna(0)
        # df = out_treat(df)
        with open("output.txt", "a") as f:
            print(f"handle_missing_values completed in {(time.time() - start_time)} seconds", file=f)
        return df
    except Exception as e:
        logger.error(f'Exeption in handle_missing_values_day:{e} ')


def capture_horizon(config, df):
    start_time = time.time()
    try:
        # df.to_csv('before_capture.csv', index=False)
        df_horizons = []
        df = df.sort_values(['product_id', 'effective_date'], ascending=[True, True])
        for h in range(1, config.get('horizon') + 1):
            df_h = df.copy()
            for i in range(0, config.get('sales_lags')):
                can_lag_features = [
                    'sales',
                ]
                for can_lag_feature in can_lag_features:
                    df_h['{}_lag_{}'.format(can_lag_feature, i + 1)] = df.groupby('product_id')[can_lag_feature].shift(
                        i + h).fillna(0)

            ##Add horizon as the variable
            df_h = df_h.assign(
                horizon=h,
            )
            # Append the df for each horizon to the list
            df_horizons.append(df_h)

        df_h = pd.concat(df_horizons)
        del df_horizons
        del df
        with open("output.txt", "a") as f:
            print(f"capture_horizon completed in {(time.time() - start_time)} seconds", file=f)
            # df_h.to_csv('after_capture.csv', index=False)
        return df_h
    except Exception as e:
        logger.error(f'Exeption in capture_horizon:{e} ')


def merge_all_level_features(df_h, df_prod_level, df_date_level):
    start_time = time.time()
    try:
        df_h = df_h.merge(df_prod_level, how='left', on='product_id')
        df_h = df_h.merge(df_date_level, how='left', on='effective_date')
        with open("output.txt", "a") as f:
            print(f"merge_all_level_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e:
        logger.error(f'Exeption in handle_missing_values:{e} ')


def add_age_in_days_feature(df_h):
    start_time = time.time()
    try:
        df_h['age_in_days'] = (df_h.effective_date - df_h.first_sale_date).dt.days
        df_h['age_in_days'] = df_h.age_in_days.where(df_h.age_in_days > 0, 0)
        with open("output.txt", "a") as f:
            print(f"add_age_in_days_feature completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e:
        logger.error(f'Exeption in merge_all_level_features:{e} ')


def add_seasonality_features(df_h):
    start_time = time.time()
    try:
        df_h = df_h.assign(
            week=df_h.effective_date.dt.week,
            month=df_h.effective_date.dt.month,
            quarter=df_h.effective_date.dt.quarter,
            day=df_h.effective_date.dt.day,
            days_in_month=df_h.effective_date.dt.daysinmonth
        )

        df_h = df_h.assign(
            week_sin=np.sin((df_h.week - 1) * (2. * np.pi / 53)),
            week_cos=np.cos((df_h.week - 1) * (2. * np.pi / 53)),
            month_sin=np.sin((df_h.month - 1) * (2. * np.pi / 12)),
            month_cos=np.cos((df_h.month - 1) * (2. * np.pi / 12)),
            quarter_sin=np.sin((df_h.quarter - 1) * (2. * np.pi / 4)),
            quarter_cos=np.cos((df_h.quarter - 1) * (2. * np.pi / 4)),
            day_sin=np.sin((df_h.day - 1) * (2. * np.pi / df_h.days_in_month)),
            day_cos=np.cos((df_h.day - 1) * (2. * np.pi / df_h.days_in_month))
        )
        with open("output.txt", "a") as f:
            print(f"add_seasonality_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e:
        logger.error(f'Exeption in add_seasonality_features {e}')


def get_Xny(config):
    start_time = time.time()
    try:

        cat_features_list = features_level_info(config, 'cat_features')
        cat_features = cat_features_list
        print(cat_features)
        can_lag_features1 = [
            'sales'
        ]
        # cont_lag_features1 = ['{}_lag_{}'.format(can_lag_feature, i+1) for can_lag_feature in can_lag_features1 for i in range(0, config.get('sales_lags'))]
        cont_lag_features1 = ['{}_lag_{}'.format(can_lag_feature, i + 1) for can_lag_feature in can_lag_features1 for i
                              in range(0, config.get('sales_lags'))]

        cont_features_others = [
            'horizon',
            'age_in_days',
            # 'weight',
            'week_sin', 'week_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'day_sin', 'day_cos'
        ]

        cont_features_holidays = ['is_holiday_fut_{}'.format(i) for i in range(config.get('futures'))]
        zero_sales_vars = ['zero_gaps_count'] + ['last_non_zero_sale']
        cont_features = cont_lag_features1 + cont_features_others + cont_features_holidays + zero_sales_vars

        less_imp_feats = [
        ]

        X = list(set(cat_features + cont_features) - set(less_imp_feats))
        print(X)
        y = ['sales']
        with open("output.txt", "a") as f:
            print(f"get_Xny completed in {(time.time() - start_time)} seconds", file=f)
        return X, y
    except Exception as e:
        logger.error('Exeption in get_Xny ', e)


def convert_cat_features_dtype(config, df_h):
    start_time = time.time()
    try:

        cat_features_list = features_level_info(config, 'cat_features')
        cat_features = cat_features_list
        for feature in cat_features:
            print(feature)
            df_h[feature] = df_h[feature].astype('category')
        with open("output.txt", "a") as f:
            print(f"convert_cat_features_dtype completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e:
        logger.error(f'Exeption in convert_cat_features_dtype:{e} ')


def get_zero_sales_vars(config,df_h):
    try:
        start_time = time.time()

        def first_nonzero(arr, axis, invalid_val=-1):
            invalid_val = arr.shape[1] - 1
            mask = arr != 0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

        cols = [f'sales_lag_{i}' for i in range(1, config.get('sales_lags')+1)]
        arr = df_h[cols].values
        df_h['zero_gaps_count'] = first_nonzero(arr, axis=1)
        df_h['last_non_zero_sale'] = arr[np.arange(len(arr)), first_nonzero(arr, axis=1)]
        with open("output.txt", "a") as f:
            print(f"get_zero_sales_vars completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e:
        logger.error(f"Exception in get zero sales vars: {e}")


def save_parauqet(config,file,file_name):
    model_path = get_root_dir()
    location = '{}/prepared_data_regression/{}/'.format(model_path, config["tenant_id"])
    Path(location).mkdir(parents=True, exist_ok=True)
    file.to_parquet(location + f'{file_name}.parquet.gzip', compression='gzip')
    return True

def prepare_initial_data_category(config,df_raw,category):

    start_time = time.time()

    try:
        if len(df_raw) == 0:
            logger.info(f'No items for category = {category}')
            with open (f"no_data_cat_{config['tenant_id']}.txt", "a") as f:
                print(category,file=f)
            return
        horizon = config.get('horizon')
        print(horizon)
        forecast_start = pd.Timestamp(config.get('infer_start_date'))
        conn = config["conn"]
        df = out_treat(df_raw)
        logger.info(f"outlier treatment for category:{category} --completed")
        horizon_dates = [forecast_start + pd.Timedelta(days=i * 7) for i in range(horizon)]
        product_ids = df.product_id.unique()
        df_f = pd.DataFrame(itertools.product(product_ids, horizon_dates), columns=['product_id', 'effective_date'])
        df = pd.concat([df, df_f])

        df_x = add_holiday_features(config, df)

        logger.info(f"added holiday features for category:{category} --completed")

        df_prod_level = get_prod_level_features(config, df_x)

        logger.info(f"get prod level features for category:{category} --completed")
        df_date_level = get_week_level_features(config, df_x)
        logger.info(f"get week level features for category:{category} --completed")
        df_prod_date_level = get_prod_week_level_features(config, df_x)

        # del df
        logger.info(f"get prod week level features for category:{category} --completed")
        df_prod_date_level = handle_missing_values(config, df_prod_date_level)
        logger.info(f"handle missing values for category:{category} --completed")
        # function for outlier treatment
        df_h = capture_horizon(config, df_prod_date_level)
        logger.info(f"capture horizon for category:{category} --completed")
        del df_prod_date_level
        df_h = merge_all_level_features(df_h, df_prod_level, df_date_level)

        del df_prod_level
        del df_date_level
        logger.info(f"merge all features for category:{category} --completed")
        df_h = add_age_in_days_feature(df_h)
        logger.info(f"add age in days for category:{category} --completed")
        df_h = add_seasonality_features(df_h)
        logger.info(f"add seasonality features for category:{category} --completed")
        df_h = convert_cat_features_dtype(config, df_h)
        #         print('======')
        logger.info(f"convert cat features for category:{category} --completed")

        ################# zero sales count and last non zero sales ###########################
        df_h = get_zero_sales_vars(config,df_h)
        df_h = df_h.drop(['ts_start','ts_end'], axis=1)
        logger.info(f" get_zero_sales_vars for category:{category} --completed")
        final_data_list = []

        train_data = df_h.query('effective_date < @forecast_start')
        for h in range(1, horizon + 1):
            test_date = forecast_start + pd.Timedelta(days=7 * (h - 1))
            infer_data = df_h.query('effective_date == @test_date and horizon==@h')
            final_data_list.append(infer_data)
        del df_h
        final_data_list.append(train_data)
        final_df = pd.concat(final_data_list)
        del train_data
        del final_data_list
        final_df["category"] = category
        with open("output.txt", "a") as f:
            print(f"**prepared_df_shape for category:{category} ={final_df.shape}** ", file=f)
        final_df["effective_date"] = pd.to_datetime(final_df["effective_date"]).dt.date
        final_df["first_sale_date"] = pd.to_datetime(final_df["first_sale_date"]).dt.date
        if config["data_save_method"] =="db":
            write_to_snowflake(df=final_df, table_name=config["prepared_data_table_name"],if_exists="append", conn=conn, add_timestamp=False)
        else:
            formatted_category = get_formatted_str(category)
            save_parauqet(config=config,file=final_df,file_name=formatted_category)
        del final_df
        with open("output.txt", "a") as f:
            print(f"prepare_initial_data_category for category:{category} completed in {(time.time() - start_time)} seconds", file=f)
        logger.info(f"  prepare_initial_data_category for category:{category} --completed")

        return True
    except Exception as e:
        logger.error(f'Exception in prepare_initial_data_category for category:{category}:{e} ')

def drop_intermediate_table_before_run(config,conn):
    drop_table_query = f""" drop table if exists {config["prepared_data_table_name"]} """
    execute_sql_on_snowflake(drop_table_query, conn)
    return True

def prepare_initial_data(config,conn,category_list):
    try:
        start_time = time.time()

        params = []
        for category in category_list:
            config["category"] = category
            raw_data = fetch_cat_raw_data(config, conn)
            config["conn"] = str(conn.url)
            params.append({"config":config,"df_raw":raw_data,"category":category})

        final_data_df_list = parallelize(function=prepare_initial_data_category, params=params, backend="loky", n_jobs =32)
        logger.info(f" #### prepare_initial_data completed for batch:{category_list}")
        with open("output.txt", "a") as f:
            print(
                f" #### prepare_initial_data completed for batch:{category_list} in {(time.time() - start_time)/60} minutes  ####",
                file=f)

    except Exception as e:
        logger.error(f"Exception in prepare_initial_data error: {e} ")
        raise ValueError("Exception in prepare_initial_data")


def fetch_prepared_data_from_source(config,category,formatted_category, conn,load_method):
    if load_method == "db":
        train_data_query = f""" select * from {config["prepared_data_table_name"]} 
                                where category = '{category}' and effective_date<='{config["end_date"]}'
                                order by product_id asc, effective_Date asc
                                 """
        infer_data_query = f""" select * from {config["prepared_data_table_name"]} 
                                where category = '{category}' and effective_date>'{config["end_date"]}'
                                order by product_id asc, effective_Date asc
                                 """
        print(train_data_query)
        train_data = pd.read_sql(train_data_query, conn)

        infer_data = pd.read_sql(infer_data_query, conn)
        train_data['effective_date'] = pd.to_datetime(train_data['effective_date'])
        infer_data['effective_date'] = pd.to_datetime(infer_data['effective_date'])
    else:
        model_path = get_root_dir()
        location = '{}/prepared_data_regression/{}/'.format(model_path, config["tenant_id"])
        full_data = pd.read_parquet(location + f'{formatted_category}.parquet.gzip')
        full_data['effective_date'] = pd.to_datetime(full_data['effective_date'])
        train_data = full_data[full_data["effective_date"] <= config["end_date"]]
        infer_data = full_data[full_data["effective_date"] > config["end_date"]]
        train_data = train_data.sort_values(by=["product_id","effective_date"])
        infer_data = infer_data.sort_values(by=["product_id", "effective_date"])

    train_data['cluster_id'] = train_data['cluster_id'].astype(int)
    infer_data['cluster_id'] = infer_data['cluster_id'].astype(int)
    train_data['product_id'] = train_data['product_id'].astype('category')
    infer_data['product_id'] = infer_data['product_id'].astype('category')

    train_data['first_sale_date'] = pd.to_datetime(train_data['first_sale_date'])
    infer_data['first_sale_date'] = pd.to_datetime(infer_data['first_sale_date'])

    return train_data, infer_data

def create_config_data(config, op_run_id, op_duration, op_dep_run_id=None):
    try:
        config_dict = dict()
        config_dict['op_name'] = config['op_name']
        config_dict['op_run_id'] = op_run_id
        config_dict['op_dependent_run_id'] = op_dep_run_id
        config_dict['tenant_id'] = config['tenant_id']
        config_dict['cluster_run_id'] = config['cluster_run_id']
        config_dict['db_name'] = config['db_name']
        config_dict['op_duration_sec'] = op_duration

        config = {key: config[key] for key in config if key not in ['tenant_id', 'cluster_run_id',
                                                                    'db_name', 'op_name']}

        config_df = pd.DataFrame(pd.Series(config_dict)).T
        config_df['op_config'] = str(config)
        config_df['updated_on'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return config_df
    except Exception as e:
        logger.error(f'Exeption in create_config_data:{e} ')


def push_config_table_to_snowflake(config, config_df, conn):
    try:
        metadata_table = CONFIG_TABLES['fore_meta_table']
        conn.execute("""USE SCHEMA {db_name}.ml""".format(db_name=config.get('db_name')))
        config_df.to_sql(metadata_table, con=conn, if_exists='append', index=False)
    except Exception as e:
        logger.error(f'Exeption in push_config_table_to_snowflake:{e} ')


def id_generator(size=10):
    try:

        return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits), size))
    except Exception as e:
        logger.error(f'Exeption in id_generator:{e} ')

