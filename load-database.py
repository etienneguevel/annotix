from argparse import ArgumentParser
from dotenv import dotenv_values
import pandas as pd
import psycopg2


def connect(database, user, password, host, port):
    return psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
    )

def query(database, schema, origin_db="Bacterial_metabolites_database"):
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

def get_arguments():
    sources = ["Bacterial_metabolites_database", "Emerging_pollutants_database", "Metabolite_database", "POS_LC", "NEG_LC", "GNPS-LIBRARY"]
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument("--sources_db", required=True, type=str, nargs="+", choices=sources)
    parser.add_argument("--to_dataframe", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()

def fetch_source(source_db, database, schema, cursor):
    msg = query(database, schema, source_db)
    cursor.execute(msg)
    row_lst = [row for row in cursor]
    print(f"{source_db:50} {len(row_lst)} lines")
    return row_lst


if __name__ == "__main__":

    args = get_arguments()

    envs = {**dotenv_values(".env")}

    database = "MolRefAnt_DB_PostGreSQL_SCAI"
    schema = "MolRefAnt_DB"

    connector = connect(
        database=database,
        user=envs.get("DB_USER"),
        password=envs.get("DB_PWD"),
        host=envs.get("DB_HOST", "localhost"),
        port=envs.get("DB_PORT", 5432)
    )
    cursor = connector.cursor()

    data_list = []
    for source_db in args.sources_db:
        data_list += fetch_source(source_db, database, schema, cursor)

    if args.to_dataframe:
        df = pd.DataFrame(data_list, columns=["spectral_data_id", "pepmass", "num_peaks", "peaks_list", "data_id", "smiles", "compound_name", 
                                              "database_name", "path_to_data", "json_file", "filename", "charge"])
        print(df[["compound_name", "pepmass", "num_peaks", "smiles", "database_name"]])

        if args.output:
            df.to_csv(args.output, index=False)

    cursor.close()
    connector.close()
