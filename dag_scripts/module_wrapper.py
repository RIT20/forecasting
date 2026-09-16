import argparse
import json
from datetime import datetime, timedelta

from core_utils.config_manager import DBManager
from core_utils.logger import logger
from src.runner import initiatePipeline, runForecastPipeline
from utils.etl import getLatestETLDate, validateETLDate, latestDataNotAvailable


def parseWrapperArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant_id", type=str, required=True)
    parser.add_argument(
        "--env_name",
        type=str,
        help="Environment Name",
        choices=["DEV", "PROD"],
        default="PROD",
    )
    parser.add_argument(
        "--action",
        type=str,
        help="Action to be run by Pipeline",
        choices=[
            "onboarding",
            "run-with-tuning",
            "run-wo-tuning",
            "run-only-tuning",
            "run-only-inference",
        ],
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="Models are trained using the data prior to end_date",
    )
    parser.add_argument(
        "--infer_start_date",
        type=str,
        help="Forecast inferences start from infer_start_date",
    )

    args = parser.parse_args()

    args_dict = args.__dict__.copy()

    return args_dict


def get_infer_start_date(dt=None):
    if dt is None:
        cur_date = datetime.utcnow()
    else:
        cur_date = datetime.strptime(dt, "%Y-%m-%d")
    if cur_date.weekday() == 6:
        start_dt = cur_date
    else:
        start_dt = cur_date - timedelta(days=cur_date.weekday() + 1)
    start_dt = start_dt.strftime("%Y-%m-%d")
    return start_dt


def get_end_date(dt=None):
    if dt is None:
        cur_date = datetime.utcnow()
    else:
        cur_date = datetime.strptime(dt, "%Y-%m-%d")
    if cur_date.weekday() == 6:
        end_dt = cur_date - timedelta(days=cur_date.weekday() + 1)
    else:
        end_dt = cur_date - timedelta(days=cur_date.weekday() + 1 + 7)
    end_dt = end_dt.strftime("%Y-%m-%d")
    return end_dt


def runForecastOperation(args: dict):
    op_name = args["op_name"]
    logger.info(f"Running Forecast Operation - {op_name}")
    if op_name == "tuning":
        end_date = args.get("end_date")
        end_date = get_end_date(dt=end_date)
        args["end_date"] = end_date

    elif op_name == "training":
        end_date = args.get("end_date")
        end_date = get_end_date(dt=end_date)
        args["end_date"] = end_date

    elif op_name == "inference":
        start_date = args.get("infer_start_date")
        start_date = get_infer_start_date(dt=start_date)
        args["infer_start_date"] = start_date

    initiatePipeline(**args)
    _ = runForecastPipeline()


def wrapper(args: dict):
    action = args["action"]
    logger.info(f"Request : {json.dumps(args)}")

    if action in [
        "onboarding",
        "run-with-tuning",
        "run-wo-tuning",
        "run-only-tuning",
        "run-only-inference",
    ]:
        end_date = (
            datetime.utcnow().strftime("%Y-%m-%d")
            if args.get("end_date") is None
            else args.get("end_date")
        )
        etl_date = getLatestETLDate()
        # flag = validateETLDate(etl_date=etl_date, date=end_date)
        flag = True
        if not flag:
            raise latestDataNotAvailable()
        args["end_date"] = min(end_date, etl_date)

        etl_date_plus1 = (
            datetime.strptime(etl_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        if (
            args.get("infer_start_date") is None
            or args.get("infer_start_date") > etl_date_plus1
        ):
            args["infer_start_date"] = etl_date_plus1

    if action == "onboarding":
        infer_start_date = datetime.strptime(
            get_infer_start_date(args["end_date"]), "%Y-%m-%d"
        ).date()
        infer_dts = [str(infer_start_date - timedelta(days=i * 7)) for i in range(5)]

        args["end_date"] = str(
            (datetime.strptime(infer_dts[-1], "%Y-%m-%d") - timedelta(days=1)).date()
        )
        args["op_name"] = "tuning"
        runForecastOperation(args)

        for infer_dt in infer_dts:
            args["end_date"] = str(
                (datetime.strptime(infer_dt, "%Y-%m-%d") - timedelta(days=1)).date()
            )
            args["op_name"] = "training"
            runForecastOperation(args)

            args["infer_start_date"] = infer_dt
            args["op_name"] = "inference"
            runForecastOperation(args)

    elif action == "run-with-tuning":
        for op_name in ["tuning", "training", "inference"]:
            args["op_name"] = op_name
            runForecastOperation(args)

    elif action == "run-wo-tuning":
        # once a week on tuesday inorder to capture data will sunday
        for op_name in ["training", "inference"]:
            args["op_name"] = op_name
            runForecastOperation(args)

    elif action == "run-only-tuning":
        # once a month any day
        for op_name in ["tuning"]:
            args["op_name"] = op_name
            runForecastOperation(args)

    elif action == "run-only-inference":
        args["op_name"] = "inference"
        runForecastOperation(args)

    return 1


if __name__ == "__main__":
    args = parseWrapperArguments()
    DBManager.set_customer(customer=args["tenant_id"], env=args["env_name"])
    _ = wrapper(args)
