import pandas as pd

from annotix_ml import ROOT


def main():
    df = pd.read_csv(
        "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym.tsv", sep="\t"
    )
    out_path = ROOT.parent / "data" / "MassSpecGym"
    out_path.mkdir(exist_ok=True, parents=True)

    print(f"saving to {out_path / 'MassSpecGym.csv'}")
    df.to_csv(out_path / "MassSpecGym.csv", index=False)


if __name__ == "__main__":
    main()
