# Import libraries
import pandas as pd
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy.dialects import registry


__author__ = "Nitesh Tripathi"


def read_from_snowflake(sql: str, sess: str) -> pd.DataFrame:
    """
    Read from Snowflake table to Pandas dataframe

    Args:
        sql: Either the path to a SQL file or the SQL query itself
        conn: DB Session

    Returns:
        data fetched using given sql.
    """
    df = pd.read_sql(sql[0], sess.bind)

    # Convert column names to lower case
    df.columns = map(str.lower, df.columns)

    return df


def write_to_snowflake(
    df: pd.DataFrame,
    table_name: str,
    conn: str,
    add_timestamp: bool = True,
    if_exists: str = "replace",
):
    """
    Write Pandas dataframe to Snowflake table

    Args:
        df: Data to write to Snowflake DB
        table_name: Table name
        conn: DB Session
        add_timestamp: Whether to add a timestamp or not
        if_exists: What to do if table already exists in DB. Can take values ['append', 'replace']
    """

    # Create copy
    table_df = df.copy()

    # Add created date
    if add_timestamp:
        table_df["created_date"] = pd.Timestamp("today")

    # Convert object type columns to string
    table_df.loc[:, table_df.dtypes == object] = table_df.loc[
        :, table_df.dtypes == object
    ].astype(str)

    # Convert column names to upper case
    table_df.columns = map(str.upper, table_df.columns)

    # Write dataframe to DB
    registry.register("snowflake", "snowflake.sqlalchemy", "dialect")
    table_df.to_sql(
        table_name, conn, if_exists=if_exists, index=False, method=pd_writer
    )


def execute_sql_on_snowflake(sql: str, conn: str) -> bool:
    """
    Executes given sql on snowflake.

    Args:
    sql -- Either the path to a SQL file or the SQL query itself
        conn: DB Session

    Returns:
        True if query is executed succesfully.
    """
    conn.execute(sql)
    # conn.commit()

    return True
