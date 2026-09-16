import os
import time
import traceback
from datetime import datetime

from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger


class latestDataNotAvailable(Exception):
    def __init__(self):
        self.message = "Latest ETL Data too old !!"

        logger.exception(self.__str__())
        raise Exception(self.__str__())

    def __str__(self):
        return self.message


def validateETLDate(etl_date, date, time_gap=14, date_format="%Y-%m-%d"):
    etl_date = datetime.strptime(etl_date, date_format)
    date = datetime.strptime(date, date_format)

    time_delta = date - etl_date
    # time_delta = abs(time_delta.days)
    time_delta = time_delta.days
    if time_delta > time_gap:
        return 0
    else:
        return 1


def getLatestETLDate():
    tenant_id = getattr(DBManager, "tenant_id")

    query = f"""select max(effective_date) as latest_etl
                from FCT_SALES where tenant_id = '{tenant_id}'"""

    results = DBManager.execute_fetch_query(query=query)

    latest_etl_date = list(results[0].values())[0]

    if type(latest_etl_date) is not str:
        latest_etl_date = latest_etl_date.strftime("%Y-%m-%d")

    return latest_etl_date


def get_etl_Pipeline(file_path1, file_path2):
    tenant_id = getattr(DBManager, "tenant_id")
    conn = getattr(DBManager, "engine")

    forecast_table = getattr(configuration.orm, "ForecastTable")
    forecast_summary_table = getattr(configuration.orm, "ForecastSummaryTable")

    try:

        t0 = time.time()

        # deleting previous data from forecasting table
        conn.execute(
            """delete from {table} where tenant_id='{tenant_id}' and forecast_type='{forecast_type}'
                                    """.format(
                table=forecast_table,
                tenant_id=tenant_id,
                forecast_type=getattr(configuration.model, "forecast_type"),
            )
        )

        file_full_path1 = os.path.abspath(file_path1)

        # TODO : to_snowflake method
        # Creating Stage table from Snowflake
        conn.execute(
            """create or replace stage ecom_stage_ml_{tenant_id}
                                    file_format = ecom_csv_format
                                """.format(
                tenant_id=tenant_id
            )
        )

        conn.execute(
            """put file://{path} @ecom_stage_ml_{tenant_id};
                                """.format(
                path=file_full_path1, tenant_id=tenant_id
            )
        )
        conn.execute(
            """copy into {table}
                                    from @ecom_stage_ml_{tenant_id}
                                    file_format = ecom_csv_format  
                                    purge = true;""".format(
                tenant_id=tenant_id, table=forecast_table
            )
        )

        file_full_path2 = os.path.abspath(file_path2)
        conn.execute(
            """put file://{path} @ecom_stage_ml_{tenant_id};
                                    """.format(
                path=file_full_path2, tenant_id=tenant_id
            )
        )
        conn.execute(
            """copy into {table}
                                    from @ecom_stage_ml_{tenant_id}  
                                    file_format = ecom_csv_format  
                                    purge = true;""".format(
                tenant_id=tenant_id, table=forecast_summary_table
            )
        )

        # Removing stage table from Snowflake
        conn.execute(
            """drop stage ecom_stage_ml_{tenant_id}
                                """.format(
                tenant_id=tenant_id
            )
        )

        logger.info(
            "Total time taken to load Forecasting output: %s",
            str(round(time.time() - t0, 3)) + " sec",
        )
    except Exception as err:
        logger.error("Found error: %s", traceback.format_exc())
