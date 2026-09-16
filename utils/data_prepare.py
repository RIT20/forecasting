import datetime 
import itertools
import os
import pathlib
import string

import holidays
import numpy as np
import pandas as pd

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.io_utils import get_sql_dir

# Setting Path
# path = os.getcwd()

# sys.path.insert(0, os.getcwd())

queryscript = pathlib.Path(os.path.join(get_sql_dir(), "feature_query.sql")).read_text(
    encoding="utf-8"
)


def features_level_info(level):
    config = configuration.model.to_dict()
    features = {
        "prod_level": [
            "cluster_id",
            "first_sale_date",
            # 'weight', 'weight_uom',
            # 'uda_level1', 'uda_level2', 'uda_level3', 'uda_level4', 'uda_level5','uda_level6',
            # 'uda_attribute1', 'uda_attribute2', 'uda_attribute3','uda_attribute4', 'uda_attribute5'
        ],
        "date_level": [
            "is_holiday_fut_{}".format(i) for i in range(config.get("futures"))
        ],
        "prod_date_level": [
            "sales",
        ],
        "cat_features": [
            "product_id",
            # 'weight_uom',
            # 'uda_level1', 'uda_level2', 'uda_level3', 'uda_level4', 'uda_level5','uda_level6',
            # 'uda_attribute1', 'uda_attribute2', 'uda_attribute3','uda_attribute4', 'uda_attribute5'
        ],
    }

    return features[level]


def get_clusterid_from_snowflake():
    tenant_id = getattr(DBManager, "tenant_id")
    conn = getattr(DBManager, "engine")

    try:
        qry = """select distinct running_id 
                from  ecom_cluster
                where   updated_on = (select max(updated_on) 
                                    from ecom_cluster)
                        and tenant_id = '{tenant_id}'"""
        qry_format = qry.format(tenant_id=tenant_id)

        cluster_run_id = conn.execute(qry_format).fetchone()[0]
        return cluster_run_id
    except Exception as e:
        logger.info("Exception in get_clusterid_from_snowflake ", e)


def fetch_raw_data():
    config = configuration.model.to_dict()

    conn = getattr(DBManager, "engine")
    tenant_id = getattr(DBManager, "tenant_id")

    try:
        if config["start_date"] is None:
            config["start_date"] = pd.Timestamp(config.get("end_date")) - pd.Timedelta(
                days=365 * 5
            )

        if config["cluster_run_id"] is None:
            # config['cluster_run_id'] = get_clusterid_from_snowflake(config, con=conn)
            config["cluster_run_id"] = 1
            configuration.setValue("model", "cluster_run_id", 1)

        end_date = pd.Timestamp(config.get("end_date")) + pd.Timedelta(days=6)
        qry = queryscript.format(
            start_date=config.get("start_date"),
            end_date=end_date,
            tenant_id=tenant_id,
            cluster_run_id=config.get("cluster_run_id"),
        )

        df = pd.read_sql(qry, conn)
        # Remove items with id null or empty string
        df = df[pd.notnull(df.product_id) & (df.product_id != "")]

        df["effective_date"] = pd.to_datetime(df["effective_date"])
        df["first_sale_date"] = pd.to_datetime(df["first_sale_date"])

        df = df.assign(cluster_id=1)
        df["cluster_id"] = df["cluster_id"].astype("int")

        #########New addition#################
        df["week_sunday"] = df["effective_date"].apply(
            lambda x: pd.Timestamp(
                (x.date() - pd.Timedelta(days=np.mod(x.day_of_week + 1, 7)))
            )
            # lambda x: pd.Timestamp((x.date() - pd.Timedelta(days = x.day_of_week  )))
        )

        df_prod = df[["product_id", "cluster_id", "first_sale_date"]].drop_duplicates(
            ["product_id"]
        )
        df_sales = (
            df.groupby(["product_id", "week_sunday"])["sales"].sum().reset_index()
        )

        # prods = df_sales.product_id.unique()
        # weeks = pd.date_range(start=df_sales.week_sunday.min(), end=df_sales.week_sunday.max(), freq='W')
        # import itertools
        # df_main = pd.DataFrame(itertools.product(prods, weeks), columns=['product_id', 'week_sunday'])

        df_main = df[["product_id", "week_sunday"]].drop_duplicates(
            ["product_id", "week_sunday"]
        )

        df_main = (
            df_main.merge(df_sales, how="left", on=["product_id", "week_sunday"])
            .fillna(0)
            .merge(df_prod, how="left", on="product_id")
        )
        df_main = df_main.rename(columns={"week_sunday": "effective_date"})

        return df_main
    except Exception as e:
        logger.info("Exception in fetch_raw_data ", e)


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    config = configuration.model.to_dict()
    try:
        us_holidays = holidays.US()

        def get_week_holiday_count(week_sunday):
            cnt = 0
            for i in range(7):
                dt = week_sunday + pd.Timedelta(days=i)
                if dt in us_holidays:
                    cnt = cnt + 1
            return cnt

        futures = config.get("futures")
        all_dates = df.effective_date.unique()
        holiday_list = []
        for dt in all_dates:
            all_holidays = dict()
            all_holidays["effective_date"] = dt

            for future in range(futures):
                dt_new = dt + pd.Timedelta(days=7 * future)
                holiday_count = get_week_holiday_count(dt_new)
                all_holidays["is_holiday_fut_{}".format(future)] = holiday_count
            holiday_list.append(all_holidays)
        all_holidays = pd.DataFrame(holiday_list)

        df = df.merge(all_holidays, how="left", on="effective_date")
        return df
    except Exception as e:
        logger.info("Exception in add_holiday_features ", e)


def get_prod_level_features(df):
    try:
        prod_level_list = features_level_info("prod_level")
        df_prod_level = df.drop_duplicates("product_id")[
            ["product_id"] + prod_level_list
        ]
        return df_prod_level
    except Exception as e:
        logger.info("Exception in get_prod_level_features ", e)


def get_week_level_features(df):
    try:
        date_level_list = features_level_info("date_level")
        df_week_level = df.drop_duplicates("effective_date")[
            ["effective_date"] + date_level_list
        ]
        return df_week_level
    except Exception as e:
        logger.info("Exception in get_week_level_features ", e)


def get_prod_week_level_features(df):
    try:
        prod_date_level_list = features_level_info("prod_date_level")
        df_prod_week_level = df[["product_id", "effective_date"] + prod_date_level_list]
        return df_prod_week_level
    except Exception as e:
        logger.info("Exception in get_prod_week_level_features ", e)


def handle_missing_values_day(df):
    try:
        ################## find the ends of each time series #############################
        df = df.assign(
            ts_start=df.groupby("product_id").effective_date.transform("min"),
            # ts_end = df.groupby('product_id').effective_date.transform('max'),
            ts_end=df.effective_date.max(),
        )

        ###################Step 1 in handling missing values ################################################
        df1 = df.drop_duplicates(["product_id"])
        idx = {}
        for i in range(df1.shape[0]):
            prod_d = df1.iloc[i]
            product_id = prod_d["product_id"]
            idx[product_id] = pd.date_range(
                start=prod_d["ts_start"], end=prod_d["ts_end"], freq="D"
            )

        df2 = pd.DataFrame(
            [(p, dt) for p, dts in idx.items() for dt in dts],
            columns=["product_id", "effective_date"],
        )
        df = df2.merge(df, how="left", on=["product_id", "effective_date"])

        del df1
        del df2

        #######################Impute the sales ##############################################
        prod_date_level_list = features_level_info("prod_date_level")
        for feat in prod_date_level_list:
            # df[feat] = df.sort_values(['product_id', 'effective_date'])[feat].fillna(method='ffill')
            df[feat] = df.sort_values(["product_id", "effective_date"])[feat].fillna(0)

        return df
    except Exception as e:
        logger.info("Exception in handle_missing_values_day ", e)


def handle_missing_values(df):
    try:
        ################## find the ends of each time series #############################
        df = df.assign(
            ts_start=df.groupby("product_id").effective_date.transform("min"),
            # ts_end = df.groupby('product_id').effective_date.transform('max'),
            ts_end=df.effective_date.max(),
        )

        ###################Step 1 in handling missing values ################################################
        df1 = df.drop_duplicates(["product_id"])
        idx = {}
        for i in range(df1.shape[0]):
            prod_d = df1.iloc[i]
            product_id = prod_d["product_id"]
            idx[product_id] = pd.date_range(
                start=prod_d["ts_start"], end=prod_d["ts_end"], freq="W-SUN"
            )

        df2 = pd.DataFrame(
            [(p, dt) for p, dts in idx.items() for dt in dts],
            columns=["product_id", "effective_date"],
        )
        df = df2.merge(df, how="left", on=["product_id", "effective_date"])

        del df1
        del df2

        #######################Impute the sales ##############################################
        prod_date_level_list = features_level_info("prod_date_level")
        for feat in prod_date_level_list:
            # df[feat] = df.sort_values(['product_id', 'effective_date'])[feat].fillna(method='ffill')
            df[feat] = df.sort_values(["product_id", "effective_date"])[feat].fillna(0)

        return df
    except Exception as e:
        logger.info("Exception in handle_missing_values_day ", e)


def capture_horizon(df):
    configs = configuration.model.to_dict()
    horizon = configs.get("horizon")
    sales_lags = configs.get("sales_lags")

    try:
        df_horizons = []
        for h in range(1, horizon + 1):
            df_h = df.sort_values(
                ["product_id", "effective_date"], ascending=[True, True]
            )
            for i in range(0, sales_lags):
                can_lag_features = ["sales"]
                for can_lag_feature in can_lag_features:
                    df_h["{}_lag_{}".format(can_lag_feature, i + 1)] = (
                        df.groupby("product_id")[can_lag_feature].shift(i + h).fillna(0)
                    )

            ##Add horizon as the variable
            df_h = df_h.assign(horizon=h)

            # Append the df for each horizon to the list
            df_horizons.append(df_h)

        df_h = pd.concat(df_horizons)

        del df_horizons
        del df

        return df_h
    except Exception as e:
        logger.info("Exception in capture_horizon ", e)


def merge_all_level_features(df_h, df_prod_level, df_date_level):
    try:
        df_h = df_h.merge(df_prod_level, how="left", on="product_id")
        df_h = df_h.merge(df_date_level, how="left", on="effective_date")
        return df_h
    except Exception as e:
        logger.info("Exception in handle_missing_values ", e)


def add_age_in_days_feature(df_h):
    try:
        df_h["age_in_days"] = (df_h.effective_date - df_h.first_sale_date).dt.days
        df_h["age_in_days"] = df_h.age_in_days.where(df_h.age_in_days > 0, 0)
        return df_h
    except Exception as e:
        logger.info("Exception in merge_all_level_features ", e)


def add_seasonality_features(df_h):
    try:
        df_h = df_h.assign(
            week=df_h.effective_date.dt.week,
            month=df_h.effective_date.dt.month,
            quarter=df_h.effective_date.dt.quarter,
            day=df_h.effective_date.dt.day,
            days_in_month=df_h.effective_date.dt.daysinmonth,
        )

        df_h = df_h.assign(
            week_sin=np.sin((df_h.week - 1) * (2.0 * np.pi / 53)),
            week_cos=np.cos((df_h.week - 1) * (2.0 * np.pi / 53)),
            month_sin=np.sin((df_h.month - 1) * (2.0 * np.pi / 12)),
            month_cos=np.cos((df_h.month - 1) * (2.0 * np.pi / 12)),
            quarter_sin=np.sin((df_h.quarter - 1) * (2.0 * np.pi / 4)),
            quarter_cos=np.cos((df_h.quarter - 1) * (2.0 * np.pi / 4)),
            day_sin=np.sin((df_h.day - 1) * (2.0 * np.pi / df_h.days_in_month)),
            day_cos=np.cos((df_h.day - 1) * (2.0 * np.pi / df_h.days_in_month)),
        )
        return df_h
    except Exception as e:
        logger.info("Exception in add_seasonality_features ", e)


def get_Xny(config):
    try:

        cat_features_list = features_level_info("cat_features")
        cat_features = cat_features_list

        can_lag_features1 = ["sales"]
        cont_lag_features1 = [
            "{}_lag_{}".format(can_lag_feature, i + 1)
            for can_lag_feature in can_lag_features1
            for i in range(0, config.get("sales_lags"))
        ]

        cont_features_others = [
            "horizon",
            "age_in_days",
            # 'weight',
            "week_sin",
            "week_cos",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "day_sin",
            "day_cos",
        ]

        cont_features_holidays = [
            "is_holiday_fut_{}".format(i) for i in range(config.get("futures"))
        ]
        cont_features = (
            cont_lag_features1 + cont_features_others + cont_features_holidays
        )

        less_imp_feats = []

        X = list(set(cat_features + cont_features) - set(less_imp_feats))
        y = ["sales"]

        return X, y
    except Exception as e:
        logger.info("Exception in get_Xny ", e)


def convert_cat_features_dtype(df_h: pd.DataFrame) -> pd.DataFrame:
    try:

        cat_features_list = features_level_info("cat_features")
        cat_features = cat_features_list
        for feature in cat_features:
            df_h[feature] = df_h[feature].astype("category")

        return df_h
    except Exception as e:
        logger.info("Exception in convert_cat_features_dtype ", e)


def fetch_prepared_data_for_inference() -> pd.DataFrame:
    config = configuration.model.to_dict()

    forecast_start = pd.Timestamp(config.get("infer_start_date"))
    horizon = config.get("horizon")
    lags = config.get("sales_lags")

    try:
        start_date = forecast_start - pd.Timedelta(days=lags * 7)
        end_date = forecast_start - pd.Timedelta(days=1 * 7)

        configuration.setValue("model", "start_date", start_date)
        configuration.setValue("model", "end_date", end_date)

        df = fetch_raw_data()

        horizon_dates = [
            forecast_start + pd.Timedelta(days=i * 7) for i in range(horizon)
        ]
        product_ids = df.product_id.unique()
        df_f = pd.DataFrame(
            itertools.product(product_ids, horizon_dates),
            columns=["product_id", "effective_date"],
        )
        df = pd.concat([df, df_f])

        df = add_holiday_features(df)
        df_prod_level = get_prod_level_features(df)
        df_date_level = get_week_level_features(df)
        df_prod_date_level = get_prod_week_level_features(df)
        del df

        df_prod_date_level = handle_missing_values(df_prod_date_level)

        df_h = capture_horizon(df_prod_date_level)
        del df_prod_date_level

        df_h = merge_all_level_features(df_h, df_prod_level, df_date_level)
        del df_prod_level
        del df_date_level

        df_h = add_age_in_days_feature(df_h)
        df_h = add_seasonality_features(df_h)
        df_h = convert_cat_features_dtype(df_h)

        test_date_begin = forecast_start
        test_data_list = []
        for h in range(1, horizon + 1):
            test_date = test_date_begin + pd.Timedelta(days=7 * (h - 1))
            test_data = df_h.query("effective_date == @test_date and horizon==@h")
            test_data_list.append(test_data)

        test = pd.concat(test_data_list)
        del test_data_list

        return test
    except Exception as e:
        logger.info("Exception in fetch_prepared_data_for_inference ", e)


def fetch_prepared_data():
    config = configuration.model.to_dict()
    try:

        df = fetch_raw_data()
        logger.info("Adding Holiday Features")
        df = add_holiday_features(df)

        logger.info("Computing Product level features")
        df_prod_level = get_prod_level_features(df)
        logger.info("Computing Week level features")
        df_date_level = get_week_level_features(df)
        logger.info("Computing Product-Week level features")
        df_prod_date_level = get_prod_week_level_features(df)
        del df

        logger.info("Imputing Product-Week level features")
        df_prod_date_level = handle_missing_values(df_prod_date_level)
        df_h = capture_horizon(df_prod_date_level)
        del df_prod_date_level

        logger.info("Merging all datasets")
        df_h = merge_all_level_features(df_h, df_prod_level, df_date_level)
        del df_prod_level
        del df_date_level

        logger.info("Adding age features")
        df_h = add_age_in_days_feature(df_h)
        logger.info("Adding seasonality features")
        df_h = add_seasonality_features(df_h)
        logger.info("Encoding categorical features")
        df_h = convert_cat_features_dtype(df_h)

        ##Remove all rows with sales 0 #############
        logger.info("Filter data with sales>0")
        df_h = df_h.query("sales>0")

        logger.info("Splitting into X and y sets")
        X, y = get_Xny(config)

        return df_h, X, y
    except Exception as e:
        logger.info("Exception in fetch_prepared_data ", e)


def create_config_data(op_run_id, op_duration, op_dep_run_id=None):
    configs = configuration.model.to_dict()
    tenant_id = getattr(DBManager, "tenant_id")

    try:
        config_dict = dict()
        config_dict["op_name"] = configs["op_name"]
        config_dict["op_run_id"] = op_run_id
        config_dict["op_dependent_run_id"] = op_dep_run_id
        config_dict["tenant_id"] = tenant_id
        config_dict["cluster_run_id"] = configs["cluster_run_id"]
        config_dict["op_duration_sec"] = op_duration

        config = {
            key: configs[key]
            for key in configs
            if key not in ["tenant_id", "cluster_run_id", "op_name"]
        }

        config_df = pd.DataFrame(pd.Series(config_dict)).T
        config_df["op_config"] = str(config)
        config_df["updated_on"] = datetime.datetime.utcnow().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return config_df
    except Exception as e:
        logger.info("Exception in create_config_data ", e)


def push_config_table_to_snowflake(config_df: pd.DataFrame):
    conn = getattr(DBManager, "engine")
    metadata_table = getattr(configuration.orm, "ForecastMetaDataTable")
    try:
        config_df.to_sql(metadata_table, con=conn, if_exists="append", index=False)
    except Exception as e:
        logger.info("Exception in push_config_table_to_snowflake ", e)


def id_generator(size=10):
    try:

        return "".join(
            np.random.choice(list(string.ascii_uppercase + string.digits), size)
        )
    except Exception as e:
        logger.info("Exception in id_generator ", e)
