import os
import sys
import pathlib
import time
import string
import pandas as pd
import numpy as np
import holidays
import itertools
import datetime

from utils.io_utils import get_sql_dir,get_root_dir
from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.hpc import parallelize
from utils.db import execute_sql_on_snowflake,write_to_snowflake
from pathlib import Path

cat_store_sql = pathlib.Path(os.path.join(get_sql_dir(), 'category_query.sql')).read_text(encoding='utf-8')
category_raw_data_sql = pathlib.Path(os.path.join(get_sql_dir(), 'feature_query_cat_do.sql')).read_text(encoding='utf-8')

def get_formatted_str(s):
    s = s.replace('/', '_')
    s = s.replace("\\", "_")
    s = s.replace(' ', '_')
    return s

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
        location = '{}/prepared_data_do/{}/'.format(model_path, config["tenant_id"])
        full_data = pd.read_parquet(location + f'{formatted_category}.parquet.gzip')
        full_data['effective_date'] = pd.to_datetime(full_data['effective_date'])
        train_data = full_data[full_data["effective_date"] <= config["end_date"]]
        infer_data = full_data[full_data["effective_date"] > config["end_date"]]
        train_data = train_data.sort_values(by=["product_id", "effective_date"])
        infer_data = infer_data.sort_values(by=["product_id", "effective_date"])

    train_data['cluster_id'] = train_data['cluster_id'].astype(int)
    infer_data['cluster_id'] = infer_data['cluster_id'].astype(int)
    train_data['product_id'] = train_data['product_id'].astype('category')
    infer_data['product_id'] = infer_data['product_id'].astype('category')


    return train_data, infer_data

def save_parauqet(config,file,file_name):
    model_path = get_root_dir()
    location = '{}/prepared_data_do/{}/'.format(model_path, config["tenant_id"])
    Path(location).mkdir(parents=True, exist_ok=True)
    file.to_parquet(location + f'{file_name}.parquet.gzip', compression='gzip')
    return True


def fetch_store_category_pairs(config, conn):
    logger.info('fetching category pairs...')
    qry = cat_store_sql.format(
        tenant_id=config.get('tenant_id'),
        db_name=config.get('db_name'),
        category_col=config.get('category_col'))
    print(qry)
    df = pd.read_sql(qry, conn)
    logger.info('Completed - fetching category pairs...')
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
            # 'weight_uom',
            # 'uda_level1', 'uda_level2', 'uda_level3', 'uda_level4', 'uda_level5','uda_level6', 
            # 'uda_attribute1', 'uda_attribute2', 'uda_attribute3','uda_attribute4', 'uda_attribute5'    
        ]
    }

    return features[level]



def fetch_cat_raw_data(config, conn):
    try:
        if config['start_date'] == None:
            config['start_date'] = pd.Timestamp(config.get('end_date')) - pd.Timedelta(days=365 * 5)

        if config['cluster_run_id'] == None:
            # config['cluster_run_id'] = get_clusterid_from_snowflake(config, con=conn)
            config['cluster_run_id'] = 1

        end_date = pd.Timestamp(config.get('end_date')) + pd.Timedelta(days=6)
        active_products_start_date = pd.Timestamp(end_date) - pd.Timedelta(days=52 * 3 * 7)
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
                category=config.get('category'),
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
        with open("output_do.txt", "a") as f:
            print(f"data fetch completed for cateogry:{config['category']} in {(time.time() - start_time)} seconds", file=f)
        with open("output_do.txt", "a") as f:
            print(f"no of timeseries in raw_data for cateogry:{config['category']} =  {len(df['product_id'].unique())} ", file=f)
        # Remove items with id null or empty string
        df = df[pd.notnull(df.product_id) & (df.product_id != '')]

        df['week_effective_date'] = pd.to_datetime(df['week_effective_date'])

        df["sales"] = df["sales"].fillna(0)
        df = df.rename(columns={'week_effective_date': 'effective_date'})
        with open("output_do.txt", "a") as f:
            print(
                f"fetch cat_raw_data completed for cateogry:{config['category']} in {(time.time() - start_time)} seconds",
                file=f)

        return df
    except Exception as e:
        logger.info(f"Exeption in fetch_cat_raw_data for cateogry:{config['category']}:{e} ")


def add_holiday_features(config, df):
    try:
        start_time = time.time()
        us_holidays = holidays.US()

        def get_week_holiday_count(week_sunday):
            cnt = 0
            for i in range(7):
                dt = week_sunday + pd.Timedelta(days=i)
                if dt in us_holidays:
                    cnt = cnt+1
            return cnt

        futures = config.get('futures')
        all_dates = df.effective_date.unique()
        holiday_list = []
        for dt in all_dates:
            all_holidays = {}
            all_holidays['effective_date'] = dt
            
            for future in range(futures):    
                dt_new = dt+pd.Timedelta(days=7*future)
                holiday_count = get_week_holiday_count(dt_new)
                all_holidays['is_holiday_fut_{}'.format(future)] = holiday_count
            holiday_list.append(all_holidays)
        all_holidays = pd.DataFrame(holiday_list)

        df = df.merge(all_holidays, how='left', on='effective_date')
        with open("output_do.txt", "a") as f:
            print(f"add_holiday_features completed in {(time.time() - start_time)} seconds", file=f)
        return df
    except Exception as e :
        logger.info('Exeption in add_holiday_features ', e)


def get_prod_level_features(config, df):
    try:
        prod_level_list = features_level_info(config, 'prod_level')
        df_prod_level = df.drop_duplicates('product_id')[['product_id'] + prod_level_list]
        return df_prod_level
    except Exception as e :
        logger.error(f'Exeption in get_prod_level_features:{e} ')

def get_week_level_features(config, df):
    try:
        date_level_list = features_level_info(config, 'date_level')
        df_week_level = df.drop_duplicates('effective_date')[['effective_date'] + date_level_list]
        return df_week_level
    except Exception as e :
        logger.error(f'Exeption in get_week_level_features:{e} ')

def get_prod_week_level_features(config, df):
    try:
        prod_date_level_list = features_level_info(config, 'prod_date_level')
        df_prod_week_level = df[['product_id', 'effective_date'] + prod_date_level_list]
        return df_prod_week_level
    except Exception as e :
        logger.error(f'Exeption in get_prod_week_level_features:{e} ')


def handle_missing_values(config, df):

    try:
        start_time = time.time()
        ################## find the ends of each time series #############################
        df = df.assign(
            ts_start = df.groupby('product_id').effective_date.transform('min'),
            # ts_end = df.groupby('product_id').effective_date.transform('max'),
            ts_end = pd.Timestamp(config["end_date"]),
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
            df[feat] = df.sort_values(['product_id', 'effective_date'])[feat].fillna(0)
        with open("output_do.txt", "a") as f:
            print(f"handle_missing_values completed in {(time.time() - start_time)} seconds", file=f)
        return df
    except Exception as e:
        logger.error(f'Exeption in handle_missing_values:{e} ')


def merge_all_level_features(df_h, df_prod_level, df_date_level):
    try:
        df_h = df_h.merge(df_prod_level, how='left', on ='product_id')
        df_h = df_h.merge(df_date_level, how='left', on ='effective_date')
        return df_h
    except Exception as e :
        logger.info('Exeption in handle_missing_values ', e)

def add_age_in_days_feature(df_h):
    try:
        df_h['age_in_days'] = (df_h.effective_date - df_h.first_sale_date).dt.days
        df_h['age_in_days'] = df_h.age_in_days.where(df_h.age_in_days>0, 0)
        return df_h
    except Exception as e :
        logger.info('Exeption in merge_all_level_features ', e)


def add_seasonality_features(df_h):
    try:
        start_time = time.time()
        df_h = df_h.assign(
            week = df_h.effective_date.dt.week,
            month = df_h.effective_date.dt.month,
            quarter = df_h.effective_date.dt.quarter,
            day = df_h.effective_date.dt.day,
            days_in_month = df_h.effective_date.dt.daysinmonth,
            dayofweek = df_h.effective_date.dt.dayofweek
        )

        df_h = df_h.assign(
            week_sin = np.sin((df_h.week-1)*(2.*np.pi/53)),
            week_cos = np.cos((df_h.week-1)*(2.*np.pi/53)),
            month_sin = np.sin((df_h.month-1)*(2.*np.pi/12)),
            month_cos = np.cos((df_h.month-1)*(2.*np.pi/12)),
            quarter_sin = np.sin((df_h.quarter-1)*(2.*np.pi/4)),
            quarter_cos = np.cos((df_h.quarter-1)*(2.*np.pi/4)),
            dayofmonth_sin = np.sin((df_h.day-1)*(2.*np.pi/df_h.days_in_month)),
            dayofmonth_cos = np.cos((df_h.day-1)*(2.*np.pi/df_h.days_in_month)),
            dayofweek_sin = np.sin((df_h.dayofweek-1)*(2.*np.pi/7)),
            dayofweek_cos = np.cos((df_h.dayofweek-1)*(2.*np.pi/7))
        )
        with open("output_do.txt", "a") as f:
            print(f"add_seasonality_features completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e :
        logger.info('Exeption in add_seasonality_features ', e)


def get_Xny(config):
    try:
        cat_features = ['product_id']
        cont_features_date_based = [
                # 'week',
                # 'month', 
                # 'quarter', 
                # 'day', 
                # 'days_in_month', 
                # 'week_sin', 'week_cos',
                # 'month_sin', 'month_cos', 
                # 'quarter_sin', 'quarter_cos',
                'horizon',
                'dayofmonth_sin', 'dayofmonth_cos',
                'dayofweek_sin', 'dayofweek_cos'
        ]
        cont_features_holidays = ['is_holiday_fut_{}'.format(i) for i in range(config.get('futures'))]
        cont_main = ['lag1', 'is_covid', 'do_in_past6', 'do1_do2', 'tp_do1', 'tp_ndo1']
        X = cont_main + cont_features_holidays + cont_features_date_based + cat_features
        y = ['sales']
        return X, y
    except Exception as e :
        logger.info(f'Exeption in get_Xny:{e} ')

def convert_cat_features_dtype(config, df_h):
    try:
        # start_time =time.time()
        cat_features = ['product_id']
        for feature in cat_features:
            df_h[feature] = df_h[feature].astype('category')
        # with open("output_do.txt", "a") as f:
        #     print(f"convert_cat_features_dtype completed in {(time.time() - start_time)} seconds", file=f)
        return df_h
    except Exception as e :
        logger.info(f'Exeption in convert_cat_features_dtype:{e} ')


def create_config_data (config, op_run_id, op_duration, op_dep_run_id=None ):
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
    except Exception as e :
        logger.info(f'Exeption in create_config_data:{e}')

    
def push_config_table_to_snowflake(config,config_df, conn):
    try:
        metadata_table = CONFIG_TABLES['fore_meta_table']
        conn.execute("""USE SCHEMA {db_name}.ml""".format(db_name = config.get('db_name')))
        config_df.to_sql(metadata_table, con=conn, if_exists='append',index=False)
    except Exception as e :
        logger.info('Exeption in push_config_table_to_snowflake ', e)



def id_generator(size=10):
    try:

        return ''.join(np.random.choice(list(string.ascii_uppercase + string.digits), size))
    except Exception as e :
        logger.info('Exeption in id_generator ', e)


def get_raw_data_with_lumpy_itmt_items(df):
    import numpy as np
    import pandas as pd
    start_time = time.time()
    start_date = df.effective_date.min()
    end_date = df.effective_date.max()

    def get_adi_cv2(df_h):
        # cols = [f'sales_lag_{i}' for i in range(1, 51)]

        cols = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
        # dt = df_h.effective_date.min() + pd.Timedelta(days = 50*7)
        # df_h = df_h.query('effective_date >= @dt')

        ##############Vectorized ADI and CV2########################
        lags = df_h[cols].values
        notnulls = (~np.isnan(lags)).astype('int')
        a1 = (lags > 0).astype('int')
        adi = np.einsum('ij -> i', notnulls) / np.einsum('ij -> i', a1)

        # mask = (lags == 0)
        mask = np.logical_or(lags == 0, np.isnan(lags))
        a2 = np.ma.MaskedArray(lags, mask)
        cv2 = ((np.std(a2, axis=1, ddof=1) / np.mean(a2, axis=1)) ** 2).data

        return adi, cv2

    dfp = pd.pivot_table(df, index='product_id', columns='effective_date', values='sales')
    adi, cv2 = get_adi_cv2(dfp)

    ts_cat = pd.DataFrame({'product_id': dfp.index, 'adi': adi, 'cv2': cv2})

    def get_category(row):
        adi = row['adi']
        cv2 = row['cv2']

        if adi < 1.32 and cv2 < 0.49:
            return 'smooth'
        elif adi >= 1.32 and cv2 < 0.49:
            return 'intermittent'
        elif adi < 1.32 and cv2 >= 0.49:
            return 'erratic'
        else:
            return 'lumpy'

    ts_cat['ts_cat'] = ts_cat.apply(lambda x: get_category(x), axis=1)
    x = ts_cat
    # x = pd.read_csv('data/ammu_tscat.csv', index_col=0)
    prods_lumpy_n_intmt = x[x['ts_cat'].isin(['lumpy', 'intermittent'])].product_id.values
    df = df.query('product_id in @prods_lumpy_n_intmt')
    with open("output_do.txt", "a") as f:
        print(f"get_raw_data_with_lumpy_itmt_items completed in {(time.time() - start_time)} seconds", file=f)
    return df

def get_main_features_old(df, start_date):

    covid_start = pd.Timestamp('2020-02-01')
    covid_end = pd.Timestamp('2022-08-01')
    # Calculating the di count in past 6
    df['do_in_past6'] = df['do'].rolling(6, min_periods=1, closed='left').sum()
    # calculating the lag
    df['lag1'] = df['do'].shift(1)
    # This drops the first timestamp
    df = df.dropna(subset=['lag1'])
    df = df.drop(['do'], axis=1)
    df = df.reset_index(drop=True)
    # counter for the first demand occurance from latest date
    demand_occurance_1 = 0
    # counter for the second demand occurance from latest date
    demand_occurance_2 = 0
    # counter for the first non demand occurance from latest date
    non_demand_occurance_1 = 0
    final_dict = {"effective_date":[], "do1_do2": [], "tp_do1": [], "tp_ndo1": []}
    for i in df.itertuples(index=False):
        if i.lag1 == 1:
            demand_occurance_2 = demand_occurance_1
            demand_occurance_1 = 0
            non_demand_occurance_1 = non_demand_occurance_1 + 1
        else:
            demand_occurance_2 = demand_occurance_2 + 1
            demand_occurance_1 = demand_occurance_1 + 1
            non_demand_occurance_1 = 0

        do1_do2 = demand_occurance_2 - demand_occurance_1
        tp_do1 = demand_occurance_1
        tp_ndo1 = non_demand_occurance_1

        final_dict["effective_date"].append(i.effective_date)
        final_dict["do1_do2"].append(do1_do2)
        final_dict["tp_do1"].append(tp_do1)
        final_dict["tp_ndo1"].append(tp_ndo1)

    df = df.assign(**final_dict)
    df = df.query(f'effective_date >= @start_date')
    df['is_covid'] = np.where(((df.effective_date >= covid_start) & ( df.effective_date <= covid_end)), 1, 0)
    return df

def get_main_features(df, start_date,horizon):
    covid_start = pd.Timestamp('2020-02-01')
    covid_end = pd.Timestamp('2022-08-01')
    df_horizons = []
    for h  in range(1, horizon+1):
        # Calculating the do count in past 6
        df_h = df.sort_values(
                ["product_id", "effective_date"], ascending=[True, True]
            )
        df_h['do_in_past6'] = df_h['do'].shift(h-1).fillna(0).rolling(6, min_periods=1, closed='left').sum()

        df_h['lag1'] = df_h['do'].shift(h).fillna(0)
        df_h = df_h.dropna(subset=['lag1'])
        df_h = df_h.drop(['do'], axis=1)
        df_h = df_h.reset_index(drop=True)
        # counter for the first demand occurance from latest date
        demand_occurance_1 = 0
        # counter for the second demand occurance from latest date
        demand_occurance_2 = 0
        # counter for the first non demand occurance from latest date
        non_demand_occurance_1 = 0
        final_dict = {"effective_date":[], "do1_do2": [], "tp_do1": [], "tp_ndo1": []}
        for i in df_h.itertuples(index=False):
            if i.lag1 == 1:
                demand_occurance_2 = demand_occurance_1
                demand_occurance_1 = 0
                non_demand_occurance_1 = non_demand_occurance_1 + 1
            else:
                demand_occurance_2 = demand_occurance_2 + 1
                demand_occurance_1 = demand_occurance_1 + 1
                non_demand_occurance_1 = 0

            do1_do2 = demand_occurance_2 - demand_occurance_1
            tp_do1 = demand_occurance_1
            tp_ndo1 = non_demand_occurance_1

            final_dict["effective_date"].append(i.effective_date)
            final_dict["do1_do2"].append(do1_do2)
            final_dict["tp_do1"].append(tp_do1)
            final_dict["tp_ndo1"].append(tp_ndo1)

        df_h = df_h.assign(**final_dict)
        df_h = df_h.assign(horizon = h)
        df_h = df_h.query(f'effective_date >= @start_date')
        df_h['is_covid'] = np.where(((df_h.effective_date >= covid_start) & ( df_h.effective_date <= covid_end)), 1, 0)
        df_horizons.append(df_h)
    final_df = pd.concat(df_horizons)

    return final_df


def get_prepared_data(df, horizon):
    try:
        start_time = time.time()
        df = df.assign(
            do=df['sales'].apply(lambda x: 1 if x > 0 else 0)
        )
        df = df.drop(['ts_start', 'ts_end', 'sales'], axis=1)
        # start_week = df.effective_date.min() + pd.Timedelta(days=40*7)
        start_week = df.effective_date.min() + pd.Timedelta(days=10 * 7 + horizon*7)
        forecast_week = df.effective_date.max() + pd.Timedelta(days=1 * 7)
        prod_ids = df.product_id.unique()
        df_2 = pd.DataFrame({"product_id": prod_ids})
        df_2["effective_date"] = forecast_week
        df_2["do"] = 0
        df = pd.concat([df, df_2])
        del df_2

        df = df.sort_values('effective_date', ascending=True)
        params = ({"df":grouped_df, "start_date":start_week, "horizon":horizon} for prod_id, grouped_df in df.groupby("product_id"))

        feature_data = parallelize(function=get_main_features,params=params,backend="serial",n_jobs=6)
        with open("output_do.txt", "a") as f:
            print(f"get_prepared_data completed in {(time.time() - start_time)} seconds", file=f)
        return pd.concat(feature_data)
    except Exception as e:
        logger.error(f"Errro in get_prepared_data:{e}")


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

        final_data_df_list = parallelize(function=prepare_initial_data_category, params=params, backend="serial", n_jobs =32)
        logger.info(f" #### prepare_initial_data completed for batch:{category_list}")
        with open("output_do.txt", "a") as f:
            print(
                f" #### prepare_initial_data completed for batch:{category_list} in {(time.time() - start_time)/60} minutes  ####",
                file=f)

    except Exception as e:
        logger.error(f"Exception in prepare_initial_data error: {e} ")
        raise ValueError("Exception in prepare_initial_data")



def prepare_initial_data_category(config,df_raw,category):
    try:
        start_time = time.time()

        conn = config["conn"]
        horizon = config.get("horizon")
        forecast_start = pd.Timestamp(config.get('infer_start_date'))
        if len(df_raw) == 0:
            logger.info(f'No items for category = {category}')
            with open (f"no_lumpy_itmt_cat_{config['tenant_id']}.txt", "a") as f:
                print(category,file=f)
            return
        df = handle_missing_values(config, df_raw)
        logger.info(f'Handling missing values completed for category:{category} --completed')

        df = get_raw_data_with_lumpy_itmt_items(df)
        if len(df) == 0:
            logger.info(f'No lumpy and intermittent items for category = {category}')
            with open (f"no_lumpy_itmt_cat_{config['tenant_id']}.txt", "a") as f:
                print(category,file=f)
            return
        logger.info(f'Fetching only lumpy and intermittent items for category:{category} --completed')

        df = get_prepared_data(df, horizon)
        if len(df) == 0:
            logger.info(f'No lumpy and intermittent items for category = {category}')
            with open (f"no_lumpy_itmt_cat_{config['tenant_id']}.txt", "a") as f:
                print(category,file=f)
            return

        logger.info(f'get_prepared_data for category:{category} --completed')


        df['cluster_id'] = 1

        # df = df_prepared

        df = df.merge(df_raw[['product_id', 'effective_date', 'sales']], how='left', on=['product_id', 'effective_date'] )
        df = df.assign(
            sales = df.sales.fillna(0)
        )
        df = add_holiday_features(config, df)
        logger.info(f"Add holiday features for category:{category} --completed")
        df = add_seasonality_features(df)
        logger.info(f"Add seasonality features for category:{category} --completed")
        df = convert_cat_features_dtype(config, df)
        logger.info(f"convert cat features for category:{category} --completed")
        cutoff = pd.Timestamp(2020, 1, 1)
        df = df.query('effective_date >= @cutoff')
        final_data_list = []

        train_data = df.query('effective_date < @forecast_start')
        test_data = df.query('effective_date >= @forecast_start')
        for h in range(1, horizon + 1):
            infer_data = test_data[test_data['horizon'] == 1]
            infer_data['effective_date'] = infer_data['effective_date'].unique()[0] + pd.DateOffset(days=7 * (h - 1))
            infer_data = infer_data.assign(horizon=h)
            final_data_list.append(infer_data)
        final_data_list.append(train_data)
        df = pd.concat(final_data_list)
        del train_data
        del test_data
        del final_data_list
        df['sales_actual'] = df.sales
        df['sales'] = df['sales_actual'].apply(lambda x : 1 if x>0 else 0)
        df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date
        df["category"] = category
        with open("output_do.txt", "a") as f:
            print(f"**prepared_df_shape for category:{category} ={df.shape}** ", file=f)
        if config["data_save_method"] == "db":
            write_to_snowflake(df=df, table_name=config["prepared_data_table_name"], if_exists="append",
                               conn=conn, add_timestamp=False)
        else:
            formatted_category = get_formatted_str(category)
            save_parauqet(config=config, file=df, file_name=formatted_category)
        del df
        with open("output_do.txt", "a") as f:
            print(f"prepare_initial_data_category for category:{category} completed in {(time.time() - start_time)} seconds", file=f)
        return True
    except Exception as e:
        logger.error(f"Exception in prepare_initial_data_category for category:{category} : {e}")


