from argparse import ArgumentParser
from dotenv import dotenv_values
import pandas as pd

from annotix_ml.load_database import connect, fetch_source


def get_arguments():
    """
    Parse command-line arguments for source databases and output options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    sources = [
        "Bacterial_metabolites_database",
        "Emerging_pollutants_database",
        "Metabolite_database",
        "POS_LC",
        "NEG_LC",
        "GNPS-LIBRARY",
    ]
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument(
        "--sources_db", required=True, type=str, nargs="+", choices=sources
    )
    parser.add_argument("--to_dataframe", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


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
        port=envs.get("DB_PORT", 5432),
    )
    cursor = connector.cursor()

    data_list = []
    for source_db in args.sources_db:
        data_list += fetch_source(source_db, database, schema, cursor)

    if args.to_dataframe:
        df = pd.DataFrame(
            data_list,
            columns=[
                "spectral_data_id",
                "pepmass",
                "num_peaks",
                "peaks_list",
                "data_id",
                "smiles",
                "compound_name",
                "database_name",
                "path_to_data",
                "json_file",
                "filename",
                "charge",
            ],
        )
        print(df[["compound_name", "pepmass", "num_peaks", "smiles", "database_name"]])

        if args.output:
            df.to_csv(args.output, index=False)

    cursor.close()
    connector.close()
