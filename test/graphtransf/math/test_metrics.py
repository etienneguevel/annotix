from annotix_ml.graphtransf.math.metrics import MCES_distance

# example_smiles = [
#     ("CC1CC(C(CC1O)O)O", "CC1=CC(C(CC1=O)O)O", 2),
#     ("C1CC2C(=O)NC(C(=O)N2C1)CC3=CC=CC=C3", "C1CC2(C(=O)NC(C(=O)N2C1)CC3=CC=CC=C3)O", 1),
#     ("CC(=O)C1=C(C2=C(C=C1O)OC(C=C2)(C)C)OC", "CC1=C(C(=C(C(=C1OC)C)O)C(=O)C)O", 7),
#     (
#         "C1=CC2=C(C(=C1)OC3C(C(C(C(O3)CO)O)O)O)C(=O)C4=C(C2C5C6=C(C(=CC=C6)OC7C(C(C(C(O7)CO)O)O)O)C(=O)C8=C5C=C(C=C8O)CO)C=C(C=C4O)CO",
#         "CC1=CC2=C(C(=C1)OC3C(C(C(C(O3)CO)O)O)O)C(=O)C4=C(C2C5C(C(C(C(O5)CO)O)O)O)C=CC=C4O",
#         30
#     ),
#     ("C1=CC=C2C(=C1)C=CC(=N2)Cl", "C1=CC2=C(C=CC(=C2)Cl)N=C1", 2),
#     ("C1CCC(=O)CCC(=O)C1", "C1CCC(=O)C1", 7),
# ]

# def test_MCES_distance():
#     # Test the MCES function on different cases
#     for smile1, smile2, expected in example_smiles:
#         mces_d = MCES_distance(smile1, smile2)

#         assert int(mces_d) == expected, f"{smile1} | {smile2}"


# if __name__ == "__main__":
#     test_MCES_distance()

