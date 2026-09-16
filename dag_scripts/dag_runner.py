import os
import sys

sys.path.insert(0, os.getcwd())

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from dag_scripts.module_wrapper import wrapper

from dag_scripts.sql_queries import (
    last_run_date_query,
    latest_etl_query,
    insert_qry,
    update_qry,
)
from core_utils.config_manager import DBManager, configuration
from core_utils.config_manager import read_yaml
from utils.io_utils import get_dag_dir

# Setting Path
from core_utils.logger import logger

# GLOBAL VARIABLES

MODULE_NAME = "ML-Forecasting"
SCHEDULER_TABLE = getattr(configuration.orm, "DagMasterScheduleTable")

SCHEDULE_INTERVALS = read_yaml(os.path.join(get_dag_dir(), "schedule_interval.yaml"))
# DB_CONNECTION = read_yaml(os.path.join(get_dag_dir(), "db_creditionals.yaml"))

DB_CONNECTION = {
    "snowflake_user": os.environ.get("USER"),
    "snowflake_password": os.environ.get("PASSWORD"),
    "snowflake_account": os.environ.get("ACCOUNT"),
    "snowflake_database": os.environ.get("DB"),
    "snowflake_schema": os.environ.get("SCHEMA"),
    "snowflake_warehouse": os.environ.get("WAREHOUSE"),
}


def updateScheduleMaster(
    tenant_id: str, jobs: list, intervals: dict = None, insert: bool = False
):
    current_date = datetime.utcnow().strftime("%Y-%m-%d")

    queries = []
    for job in jobs:
        if insert:
            query = insert_qry.format(
                table=SCHEDULER_TABLE,
                tenantid=tenant_id,
                job=job,
                module=MODULE_NAME,
                interval=intervals.get(job, "NULL"),
                curr_date=current_date,
            )
        else:
            query = update_qry.format(
                table=SCHEDULER_TABLE,
                tenantid=tenant_id,
                jobname=job,
                module=MODULE_NAME,
                curr_date=current_date,
            )

        queries.append(query)

    conn = getattr(DBManager, "engine")
    for q in queries:
        conn.execute(q)


def isOnboarding(tenant_id):
    conn = getattr(DBManager, "engine")

    # Check if tenant_id is present in DAG Schedule Master Table
    qry = last_run_date_query.format(
        table=SCHEDULER_TABLE,
        tenantid=tenant_id,
        module=MODULE_NAME,
        jobname="onboarding",
    )
    results = conn.execute(qry).fetchone()

    if not results:
        return True
    else:
        # Check if there is new inbound data
        last_run_date = results[0]
        latest_etl_date = conn.execute(
            latest_etl_query.format(tenantid=tenant_id)
        ).fetchone()[0]

        if last_run_date < latest_etl_date:
            return True
        else:
            return False


def getScheduleJobs(tenant_id):
    conn = getattr(DBManager, "engine")

    schedule_jobs = []
    for job in list(SCHEDULE_INTERVALS.keys()):
        qry = last_run_date_query.format(
            table=SCHEDULER_TABLE,
            tenantid=tenant_id,
            jobname=job,
            module=MODULE_NAME,
        )
        last_run_date = conn.execute(qry).fetchone()[0]
        last_run_date = last_run_date.date()
        current_date = datetime.utcnow().date()

        if (current_date - last_run_date).days >= SCHEDULE_INTERVALS[job]:
            schedule_jobs.append(job)
        else:
            print("{} Module- {} is already updated".format(MODULE_NAME, job))

    return schedule_jobs


def run_dag(request: dict):
    tenant_id = request["tenant_id"]

    DB_CONNECTION["tenant_id"] = tenant_id
    request["connection"] = DB_CONNECTION

    DBManager.set_customer(connection=DB_CONNECTION)

    onboarding_flag = isOnboarding(tenant_id=tenant_id)
    if onboarding_flag:
        request["action"] = "onboarding"
        print(request)
        logger.info(f"Executing {request['action']} action")
        _ = wrapper(args=request)

        updateScheduleMaster(
            tenant_id=tenant_id,
            jobs=["onboarding", "run-only-tuning", "run-wo-tuning"],
            intervals=SCHEDULE_INTERVALS,
            insert=True,
        )
    else:
        jobs = getScheduleJobs(tenant_id=tenant_id)
        for job in jobs:
            request["action"] = job
            logger.info(f"Executing {request['action']} action")
            _ = wrapper(args=request)
            updateScheduleMaster(tenant_id=tenant_id, jobs=[job])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant_id", type=str, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    args = parser.parse_args()
    request = args.__dict__.copy()

    run_dag(request)
