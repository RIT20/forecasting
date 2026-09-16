import os


def get_root_dir() -> str:
    """
    Returns the project root directory
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(curr_dir, "../"))

def get_dag_dir() -> str:
    """
    Returns the dag config directory
    """
    return os.path.join(get_root_dir(), "dag_scripts")

def get_config_dir() -> str:
    """
    Returns the config directory
    """
    return os.path.join(get_root_dir(), "config")


def get_src_dir() -> str:
    """
    Returns the config directory
    """
    return os.path.join(get_root_dir(), "src")


def get_data_dir() -> str:
    """
    Returns the data directory
    """
    return os.path.join(get_root_dir(), "data_io")


def get_sql_dir() -> str:
    """
    Returns the data directory
    """
    return os.path.join(get_root_dir(), "sql")

def get_model_dir() -> str:
    """
    Returns the models directory
    """
    return os.path.join(get_root_dir(), "model_save")


def get_log_dir() -> str:
    """
    Returns the log directory
    """
    # Locate directory to log
    base_log_dir = "/var/logs/"
    local_log_dir = os.path.join(get_root_dir(), "logs")

    # If the current user running does not have access to log to the above directory, then log it local folder
    if os.access(base_log_dir, os.W_OK):
        return base_log_dir
    else:
        os.makedirs(local_log_dir, exist_ok=True)
        return local_log_dir
