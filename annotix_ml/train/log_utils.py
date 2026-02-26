import wandb
from rdkit import Chem
from rdkit.Chem import Draw


def create_gen_samples_table(gen_metrics):
    """
    Create a wandb Table for generated samples with SMILES, molecule images, and metrics.

    Args:
        gen_metrics (dict): Dictionary containing true_smiles, all_gen_smiles, mces, and tan_sim.

    Returns:
        wandb.Table: The populated wandb Table.
    """
    columns = [
        "true_smiles",
        "true_mol",
        "gen_smiles",
        "gen_mol",
        "MCES",
        "tanimoto_sim",
    ]
    table = wandb.Table(columns=columns)

    true_smiles = gen_metrics.get("true_smiles", [])
    gen_smiles = gen_metrics.get("all_gen_smiles", [])
    mces = gen_metrics.get("mces", [])
    tanimoto = gen_metrics.get("tan_sim", [])

    for ts, gs, m, t in zip(true_smiles, gen_smiles, mces, tanimoto):
        true_mol_img = None
        if ts:
            mol = Chem.MolFromSmiles(ts)
            if mol:
                true_mol_img = wandb.Image(Draw.MolToImage(mol))

        gen_mol_img = None
        if gs:
            mol = Chem.MolFromSmiles(gs)
            if mol:
                gen_mol_img = wandb.Image(Draw.MolToImage(mol))

        table.add_data(ts, true_mol_img, gs, gen_mol_img, m, t)

    return table
