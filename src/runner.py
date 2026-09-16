import src.inference as inference
import src.training as training
import src.tuning as tuning
from core_utils.config_manager import DBManager, configuration
from core_utils.logger import logger
from utils.argparser import parseArguments
from utils.dict_utils import merge_default_dict


def initiatePipeline(**kwargs):
    """
    Initiates the Model Pipeline by parsing and setting the configurations for the session
    """

    tenant_id = kwargs["tenant_id"]
    env_name = kwargs.get("env_name")
    connection = kwargs.pop("connection", None)

    logger.info(f"Initializing the Pipeline")
    if connection:
        logger.info(f"Customer Name : {tenant_id}")
        DBManager.set_customer(connection=connection)
    else:
        logger.info(f"Customer Name : {tenant_id}")
        logger.info(f"Environment : {env_name}")
        DBManager.set_customer(customer=tenant_id, env=env_name)

    # Load configuration files
    config_load_type = "LOCAL"
    logger.info(f"Loading Configurations from {config_load_type}")
    configuration.parseConfigs(parse=config_load_type)

    # Merge command line arguments with model configurations
    model_configs = configuration.model.to_dict()
    merged_config = merge_default_dict(kwargs, model_configs)

    configuration.addConfigs(name="model", value=merged_config)


def runForecastPipeline():
    logger.info(f"Initiated Forecast Pipeline")
    configs = configuration.model.to_dict()
    if configs["op_name"] == "tuning":
        tuning.tune_for()
    elif configs["op_name"] == "training":
        training.train_for()
    elif configs["op_name"] == "inference":
        inference.infer_for()
    else:
        logger.info(
            " Incorrect Operation name ! Please select tuning, training or inference"
        )
        logger.error("Error in operation name...")


if __name__ == "__main__":
    args = parseArguments()

    initiatePipeline(**args)

    runForecastPipeline()

    # conn = getattr(DBManager, "engine")
    # forecast_table = getattr(configuration.orm, "ForecastTable")
