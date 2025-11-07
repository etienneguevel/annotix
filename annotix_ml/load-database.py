"""
Fetch spectral and compound data from a PostgreSQL database.

This script connects to a specified database, queries for spectral data from selected sources,
and optionally outputs the results as a pandas DataFrame or CSV file.
"""

import psycopg2


def connect(database, user, password, host, port):
    """
    Establish a connection to a PostgreSQL database.

    Args:
        database (str): Database name.
        user (str): Username.
        password (str): Password.
        host (str): Host address.
        port (int): Port number.

    Returns:
        connection: psycopg2 connection object.
    """
    return psycopg2.connect(
        database=database, user=user, password=password, host=host, port=port
    )


def query(database, schema, origin_db="Bacterial_metabolites_database"):
    """
    Generate a SQL query to fetch spectral and compound data for a given source.

    Args:
        database (str): Database name.
        schema (str): Schema name.
        origin_db (str): Source database name.

    Returns:
        str: SQL query string.
    """
    return f"""select distinct
                S.spectral_data_id, S.pepmass, S.num_peaks, S.peaks_list, S.data_id,
                C.smiles,  C.compound_name,
                D.database_name,
                DD.path_to_data, DD.json_file,
                A.filename,
                CH.charge
            from "{database}"."{schema}".spectral_data S
                join "{database}"."{schema}".compound C on S.spectral_data_id = C.spectral_data_id
                join "{database}"."{schema}".database_details D on S.database_id = D.database_id
                join "{database}"."{schema}".datetable T on T.date_id = S.data_id
                join "{database}"."{schema}".data DD on DD.data_id = S.data_id
                join "{database}"."{schema}".analytics_data A on T.analytics_data_id = A.analytics_data_id
                join "{database}"."{schema}".charge CH on S.charge_id = CH.charge_id
                where D.database_name='{origin_db}' and S.pepmass!=999.9999;"""


def fetch_source(source_db, database, schema, cursor):
    """
    Execute a query for a given source database and fetch results.

    Args:
        source_db (str): Source database name.
        database (str): Database name.
        schema (str): Schema name.
        cursor: psycopg2 cursor object.

    Returns:
        list: List of rows fetched from the database.
    """
    msg = query(database, schema, source_db)
    cursor.execute(msg)
    row_lst = [row for row in cursor]
    print(f"{source_db:50} {len(row_lst)} lines")
    return row_lst
