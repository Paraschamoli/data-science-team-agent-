"""SQL database interaction utilities."""

from typing import Any

import pandas as pd
import sqlalchemy as sql
from langchain.tools import tool  # type: ignore[import]


@tool
def execute_sql_query(
    query: str,
    connection_string: str,
    max_rows: int = 1000,
) -> tuple[str, dict[str, Any]]:
    """Execute a SQL query and return results.

    Parameters
    ----------
    query : str
        SQL query to execute
    connection_string : str
        Database connection string
    max_rows : int, optional
        Maximum number of rows to return. Defaults to 1000.

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and results

    """
    print("    * Tool: execute_sql_query")

    try:
        # Create database engine
        engine = sql.create_engine(connection_string)

        # Execute query
        df = pd.read_sql(query, engine, chunksize=max_rows)

        # Convert to list if it's a TextFileReader (for chunked queries)
        if hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
            df = next(df)  # Get first chunk

        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)

        message = f"Query executed successfully. Returned {len(df)} rows."

        return message, {"data": df.to_dict(), "shape": df.shape, "columns": list(df.columns), "query": query}

    except Exception as e:
        error_message = f"Error executing SQL query: {e!s}"
        return error_message, {"error": error_message}


@tool
def get_table_schema(
    table_name: str,
    connection_string: str,
) -> tuple[str, dict[str, Any]]:
    """Get schema information for a database table.

    Parameters
    ----------
    table_name : str
        Name of the table
    connection_string : str
        Database connection string

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and schema information

    """
    print(f"    * Tool: get_table_schema | {table_name}")

    try:
        engine = sql.create_engine(connection_string)

        # Get schema information (this varies by database type)
        if "sqlite" in connection_string.lower():
            schema_query = f"PRAGMA table_info({table_name})"
        elif "postgresql" in connection_string.lower() or "mysql" in connection_string.lower():
            schema_query = f"""  # noqa: S608 - controlled table name
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """  # noqa: S608 - controlled table name
        else:
            # Generic approach
            schema_query = f"SELECT * FROM {table_name} LIMIT 1"  # noqa: S608 - controlled table name

        df = pd.read_sql(schema_query, engine)

        message = f"Schema retrieved for table: {table_name}"

        return message, {
            "table_name": table_name,
            "schema": df.to_dict(),
            "columns": list(df.columns) if not df.empty else [],
        }

    except Exception as e:
        error_message = f"Error getting table schema: {e!s}"
        return error_message, {"error": error_message}


@tool
def list_database_tables(
    connection_string: str,
) -> tuple[str, dict[str, Any]]:
    """List all tables in the database.

    Parameters
    ----------
    connection_string : str
        Database connection string

    Returns
    -------
    Tuple[str, Dict[str, Any]]
        Tuple containing message and list of tables

    """
    print("    * Tool: list_database_tables")

    try:
        engine = sql.create_engine(connection_string)

        # Query to list tables (varies by database type)
        if "sqlite" in connection_string.lower():
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif "postgresql" in connection_string.lower():
            tables_query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        elif "mysql" in connection_string.lower():
            tables_query = "SHOW TABLES"
        else:
            # Generic approach
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"

        df = pd.read_sql(tables_query, engine)

        # Extract table names
        if "sqlite" in connection_string.lower():
            tables = df["name"].tolist()
        elif "mysql" in connection_string.lower():
            tables = df.iloc[:, 0].tolist()  # First column
        else:
            tables = df["tablename"].tolist()

        message = f"Found {len(tables)} tables in database"

        return message, {"tables": tables, "count": len(tables)}

    except Exception as e:
        error_message = f"Error listing database tables: {e!s}"
        return error_message, {"error": error_message}
