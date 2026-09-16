import argparse

from core_utils.config_manager import configuration


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant_id", type=str, required=True)
    parser.add_argument(
        "--op_name",
        type=str,
        help="Operation",
        choices=["tuning", "training", "inference"],
        default="tuning",
    )
    parser.add_argument("--start_date", type=str, help="Start Date")
    parser.add_argument("--end_date", type=str, help="End Date")
    parser.add_argument("--infer_start_date", type=str, help="Infer Start Date")
    parser.add_argument("--cluster_run_id", type=str, help="Cluster Run ID")
    parser.add_argument("--n_jobs", type=int, help="Number of Parallel Jobs")
    parser.add_argument(
        "--env_name",
        type=str,
        help="Environment Name",
        choices=["DEV", "PROD"],
        default="PROD",
    )
    parser.add_argument(
        "--horizon",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "horizon"),
        type=int,
    )
    parser.add_argument(
        "--tune_method",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "tune_method"),
        type=str,
    )
    parser.add_argument(
        "--sales_lags",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "sales_lags"),
        type=int,
    )
    parser.add_argument(
        "--loc_count_lags",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "loc_count_lags"),
        type=int,
    )
    parser.add_argument(
        "--futures",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "futures"),
        type=int,
    )
    parser.add_argument(
        "--tuning_n_trials",
        nargs="?",
        const=1,
        default=getattr(configuration.model, "tuning_n_trials"),
        type=int,
    )

    args = parser.parse_args()

    args_dict = args.__dict__.copy()

    return args_dict
