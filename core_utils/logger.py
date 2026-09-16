import logging
import logging.config
import os

#logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger('snowflake.connector').setLevel(logging.WARNING)

from utils.io_utils import get_config_dir, get_log_dir

# Locate directory to log
log_dir = get_log_dir()
path_to_log = os.path.join(log_dir, "forecasting_ml.log")

config_file = os.path.join(get_config_dir(), "logging_config.ini")
logging.config.fileConfig(config_file, defaults={"logfilename": path_to_log})

logger = logging.getLogger(__name__)
