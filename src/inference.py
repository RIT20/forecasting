import itertools
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.data_prepare import fetch_prepared_data_for_inference, id_generator
from utils.data_prepare import push_config_table_to_snowflake, create_config_data
from utils.etl import get_etl_Pipeline
from utils.io_utils import get_model_dir


# sys.path.insert(0, os.getcwd())


def load_model(train_run_id, cluster_id, quantile):
    tenant_id = getattr(DBManager, "tenant_id")
    try:
        model_path = get_model_dir()
        model = lgb.Booster(
            model_file="{}training/{}/{}/model_{}_{}.txt".format(
                model_path, tenant_id, train_run_id, cluster_id, quantile
            )
        )
        return model
    except Exception as e:
        logger.info("Exception in load_model ", e)


def save_data_and_push_to_snowflake(
    out_df1: pd.DataFrame, out_df2: pd.DataFrame, infer_run_id: int
):
    tenant_id = getattr(DBManager, "tenant_id")

    try:
        model_path = get_model_dir()
        location = "{}forecasts/{}/{}".format(model_path, tenant_id, infer_run_id)
        Path(location).mkdir(parents=True, exist_ok=True)
        out_df1.to_csv(location + "/out1.csv", index=False)
        file_path1 = location + "/out1.csv"

        out_df2.to_csv(location + "/out2.csv", index=False)
        file_path2 = location + "/out2.csv"

        get_etl_Pipeline(file_path1, file_path2)
    except Exception as e:
        logger.info("Exception in save_data ", e)


def get_train_runid_from_snowflake() -> int:
    conn = getattr(DBManager, "engine")
    tenant_id = getattr(DBManager, "tenant_id")

    metadata_table = getattr(configuration.orm, "ForecastMetaDataTable")

    try:
        qry = """select op_run_id 
                from  {table}
                where   updated_on = (select max(updated_on) 
                                    from {table}
                                    where  op_name = 'training')
                        and tenant_id = '{tenant_id}'
                        and op_name = 'training'"""
        qry_format = qry.format(tenant_id=tenant_id, table=metadata_table)

        train_runid = conn.execute(qry_format).fetchone()[0]
        return train_runid
    except Exception as e:
        logger.info("Exception in get_train_runid_from_snowflake ", e)


def disaggregate(df: pd.DataFrame) -> pd.DataFrame:
    conn = getattr(DBManager, "engine")
    tenant_id = getattr(DBManager, "tenant_id")

    config = configuration.model.to_dict()

    params = {
        "tenant_id": tenant_id,
        "infer_start_sunday": config.get("infer_start_date"),
        "disagg_consideration_window": config.get("sales_lags"),
    }

    qry = """
    with t1 as
    (
    select 
        -- item ,
        item || '$#$' || loc || '$#$' || channel as item,
        MOD(DAYOFWEEK(effective_date)+1, 7) as dayofweek,
        quantity
    from
        (
        SELECT * 
        FROM fct_sales
        WHERE tenant_id = '{tenant_id}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY tenant_id, item, loc, channel, effective_date ORDER BY quantity desc) = 1
        )
    where effective_date >= dateadd(week,-{disagg_consideration_window}, to_date('{infer_start_sunday}')) and  effective_date < to_date('{infer_start_sunday}')
    ) 

    select
        item,
        dayofweek,
        avg(quantity) as avg_sales,
        sum(quantity) as total_sales,
        count(dayofweek) as count_days_data_avl
    from t1
    group by item, dayofweek
    """.format(
        **params
    )

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

    df["forecast_type"] = getattr(configuration.model, "forecast_type")
    df["duration"] = 1
    df["quantity_old"] = df["quantity"]
    df["quantity"] = df["quantity"] * df["weight"]
    df = df.drop(columns=["effective_date_old", "week_sunday", "weight", "dayofweek"])

    return df


def get_outputs(df: pd.DataFrame):
    infer_start_date = getattr(configuration.model, "infer_start_date", None)

    out1 = df.copy()
    out1 = out1.assign(loc=None, channel=None)
    out1 = out1.query("quantile == 0.5")
    out1_cols_order = [
        "item",
        "loc",
        "channel",
        "tenant_id",
        "forecast_type",
        "duration",
        "quantity",
        "effective_date",
        "created_at",
    ]
    df1 = out1["item"].str.split(pat="\$#\$", expand=True)
    out1["item"] = df1[0]
    out1["loc"] = df1[1]
    out1["channel"] = df1[2]
    out1 = out1[out1_cols_order]

    out2_list = []
    for q in [0.5, 0.05, 0.95]:
        out2 = df.query("quantile == @q")
        if q == 0.5:
            out2 = out2.assign(measure_name="quantity")
        if q == 0.05:
            out2 = out2.assign(measure_name="quantity_lower")
        if q == 0.95:
            out2 = out2.assign(measure_name="quantity_upper")

        df1 = out2["item"].str.split(pat="\$#\$", expand=True)
        out2["item"] = df1[0]
        out2["loc"] = df1[1]
        out2["channel"] = df1[2]
        out2 = out2.assign(
            consumer_id=None,
            forecast_frequency_type="daily",
            forecast_at_date=infer_start_date,
            measure_dist=None,
            forecast_engine_run_timestamp=pd.Timestamp.now(),
            user_id=None,
            updated_ts=pd.Timestamp.now(),
            run_id=0,
        )
        out2["forecast_for_date"] = out2["effective_date"]
        out2["measure_value"] = out2["quantity"]
        out2_list.append(out2)

    out2 = pd.concat(out2_list)
    out2_cols_order = [
        "tenant_id",
        "item",
        "loc",
        "channel",
        "consumer_id",
        "forecast_frequency_type",
        "forecast_at_date",
        "forecast_for_date",
        "measure_name",
        "measure_value",
        "measure_dist",
        "forecast_engine_run_timestamp",
        "user_id",
        "infer_run_id",
        "train_run_id",
        "cluster_run_id",
        "updated_ts",
        "run_id",
    ]
    out2 = out2[out2_cols_order]
    return out1, out2


def infer_for():
    try:
        t0 = time.time()

        infer_run_id = id_generator()

        logger.info(f"Fetching Data for inference")
        df_h = fetch_prepared_data_for_inference()

        train_run_id = get_train_runid_from_snowflake()

        outs = []
        for cluster_id in df_h.cluster_id.unique():
            try:
                for quantile in [0.05, 0.5, 0.95]:
                    cluster_id = int(cluster_id)
                    df_h_c = df_h.query("cluster_id == @cluster_id")
                    logger.info(f"Loading the trained model")
                    model = load_model(train_run_id, cluster_id, quantile)
                    X = model.feature_name()
                    logger.info(f"Computing forecast inferences")
                    pred_results = df_h_c.assign(pred_sales=model.predict(df_h_c[X]))
                    out = pred_results[["product_id", "effective_date", "pred_sales"]]
                    out["quantile"] = quantile
                    outs.append(out)
            except Exception as e:
                print("Could not train for cluster: {}".format(cluster_id), e)
                continue

        out = pd.concat(outs)
        out = out.assign(
            tenant_id=getattr(DBManager, "tenant_id"),
            infer_run_id=infer_run_id,
            train_run_id=train_run_id,
            cluster_run_id=getattr(configuration.model, "cluster_run_id"),
            forecast_type=getattr(configuration.model, "forecast_type"),
            duration=7,
            created_at=pd.Timestamp.now(),
        ).rename(columns={"product_id": "item", "pred_sales": "quantity"})

        cols_order = [
            "item",
            "tenant_id",
            "forecast_type",
            "duration",
            "quantity",
            "effective_date",
            "infer_run_id",
            "train_run_id",
            "cluster_run_id",
            "created_at",
            "quantile",
        ]
        out = out[cols_order]

        logger.info(f"Post processing the inferences")
        out = disaggregate(out)
        out = out[cols_order]

        out1, out2 = get_outputs(out)
        logger.info(f"Export forecasts to Database")
        save_data_and_push_to_snowflake(out1, out2, infer_run_id)

        logger.info(f"Save execution log")
        op_duration_sec = round(time.time() - t0, 3)
        config_df = create_config_data(infer_run_id, op_duration_sec, train_run_id)
        push_config_table_to_snowflake(config_df)
    except Exception as err:
        logger.error("Error in infer_for: %s", traceback.format_exc())
