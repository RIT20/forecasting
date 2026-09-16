import os
import warnings
from ast import literal_eval
from collections import defaultdict
from typing import Dict, List

import yaml
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeferredReflection
from sqlalchemy.orm import Session

from utils.dict_utils import merge_default_dict
from utils.io_utils import get_config_dir

warnings.filterwarnings("ignore")

# Define path relative to this file
config_dir = get_config_dir()


def convert_type(s):
    try:
        return literal_eval(s)
    except Exception as err:
        return s


def read_yaml(file_name):
    with open(file_name, "r") as yamlfile:
        contents = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return contents


class ParserClass(object):
    def __init__(self, config: Dict):
        self.set_parameters(config)

    def set_parameters(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                setattr(self, k, ParserClass(v))
            else:
                setattr(self, k, v)

    def to_dict(self, obj=None):
        if obj is None:
            obj = self

        output = obj.__dict__.copy()
        for key, val in output.items():
            if isinstance(val, ParserClass):
                output[key] = self.to_dict(obj=val)

        return output

    def get(self, name):
        if not hasattr(self, name):
            return None
        else:
            return getattr(self, name)


class ConfigLoader(object):
    """
    Wrapper class to parser and store configurations
    """

    def __init__(self):
        self.config_files = {
            "model": os.path.join(config_dir, "model_params.yaml"),
            "orm": os.path.join(config_dir, "datamodel_config.yaml"),
        }

        self.model = self.parseConfigFile(self.config_files["model"])
        self.orm = self.parseConfigFile(self.config_files["orm"])

    def parseConfigFile(self, config_file):
        return ParserClass(config=self.readYAML(config_file))

    @staticmethod
    def readYAML(file_name):
        with open(file_name, "r") as yamlfile:
            contents = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return contents

    @staticmethod
    def imputeAttrValue(value_type, attr_value):
        if value_type in ["LIST", "JSON"]:
            return eval(attr_value)
        elif value_type in ["FLOAT"]:
            return float(attr_value)
        elif value_type in ["INTEGER"]:
            return int(attr_value)
        else:
            return attr_value

    @staticmethod
    def addConfigs(name: str, value: dict):
        setattr(configuration, name, ParserClass(value))

    def setValue(self, group, attr_name, attr_value):
        config = getattr(self, group)
        setattr(config, attr_name, attr_value)
        setattr(self, group, config)

    def parseConfigs(self, parse="LOCAL"):
        """
        Args:
            parse : LOCAL/DB
                    LOCAL - parses the default configs from local
                    DB - parses the configs from database
        """

        if parse == "LOCAL":
            pass
        elif parse == "DATABASE":
            config_table = getattr(self.orm, "ElasticityModelConfig")
            tenant_id = getattr(DBManager, "tenant_id")

            query = f"select * from {config_table} where tenant_id = '{tenant_id}'"
            db_configs = DBManager.execute_fetch_query(query)

            config_dict = defaultdict(dict)
            for item in db_configs:
                group_attr_name = item.get("attr_group")
                attr_name = item.get("attr_name")
                attr_value = item.get("attr_value")

                if attr_value:
                    attr_value = convert_type(attr_value)
                else:
                    attr_value = None

                if (group_attr_name is None) or (group_attr_name.strip() == "NULL"):
                    config_dict[attr_name] = attr_value
                else:
                    config_dict[group_attr_name][attr_name] = attr_value

            # Update Configs
            default_configs = getattr(self, "model").to_dict()
            merged_dict = merge_default_dict(config_dict, default_configs)
            setattr(self, "model", ParserClass(merged_dict))

        else:
            raise Exception("Invalid parse parameter provided !!")


class DBManager:
    base = declarative_base(cls=DeferredReflection)

    config_file = os.path.join(config_dir, "snowflake_config.yaml")

    __conf = read_yaml(config_file)

    @staticmethod
    def config(name):
        return DBManager.__conf[name]

    @staticmethod
    def set(name, value):
        setattr(DBManager, name, value)

    @staticmethod
    def set_customer(customer=None, connection=None, env="DEV"):
        if connection:
            url = URL(
                user=connection["snowflake_user"],
                password=connection["snowflake_password"],
                account=connection["snowflake_account"],
                warehouse=connection["snowflake_warehouse"],
                database=connection["snowflake_database"],
                schema=connection["snowflake_schema"],
                client_session_keep_alive=True
            )
            engine = create_engine(url,pool_size=10, max_overflow=25, echo=False)
            customer = connection["snowflake_database"]
            tenant_id = connection["tenant_id"]

        elif customer:
            engine = DBManager.get_snowflake_engine(customer, env=env)
            tenant_id = customer
        else:
            raise Exception(
                "Either of the arguments customer / connection should be provided !!"
            )

        DBManager.set("customer", customer)
        DBManager.set("tenant_id", tenant_id)
        DBManager.set("engine", engine)
        DBManager.set("session", Session(engine))

        if not getattr(configuration, "orm").get("use_cache"):
            engine.execute("ALTER SESSION SET USE_CACHED_RESULT = FALSE")

    @staticmethod
    def get_warehouse(customer: str, env: str):
        if env == "DEV":
            return "DEV_BI"
        else:
            return "PROD_ECOM_OUTPUT"

    @staticmethod
    def get_snowflake_engine(customer: str, env="DEV", echo=False):
        url = URL(
            user=DBManager.config(f"DATABASE_USER_{env}"),
            password=DBManager.config(f"DATABASE_PASSWORD_{env}"),
            account=DBManager.config(f"DATABASE_ACCOUNT_{env}"),
            warehouse=DBManager.get_warehouse(customer=customer, env=env),
            database=DBManager.config(f"DATABASE_NAME_{env}"),
            schema=DBManager.config(f"DATABASE_SCHEMA_{env}"),
            client_session_keep_alive=True
        )
        engine = create_engine(url,pool_size=10, max_overflow=25, echo=echo)

        return engine

    @staticmethod
    def execute_fetch_query(query) -> List[Dict]:
        """
        Execute and fetch output of given query
        """

        results = []

        with getattr(DBManager, "session") as sess:
            cursor = sess.execute(query)
            col_names = cursor.keys()
            for row in cursor.fetchall():
                results.append(dict(zip(col_names, row)))

        return results


configuration = ConfigLoader()

if __name__ == "__main__":
    print("Success !!")
