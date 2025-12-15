from collections import namedtuple

from rdkit import Chem


Atom = namedtuple("atom", ["weight", "covalence"])

VALID_ELEMENTS = {
    "C": Atom(weight=12.011, covalence=4),
    "As": Atom(weight=74.9216, covalence=3),
    "B": Atom(weight=10.811, covalence=3),
    "Br": Atom(weight=79.904, covalence=1),
    "Cl": Atom(weight=35.453, covalence=1),
    "Co": Atom(weight=58.933195, covalence=2),
    "F": Atom(weight=18.998403, covalence=1),
    "Fe": Atom(weight=55.845, covalence=2),
    "I": Atom(weight=126.90447, covalence=1),
    "K": Atom(weight=39.0983, covalence=1),
    "N": Atom(weight=14.007, covalence=3),
    "Na": Atom(weight=22.989769, covalence=1),
    "O": Atom(weight=15.999, covalence=2),
    "P": Atom(weight=30.973762, covalence=3),
    "S": Atom(weight=32.06, covalence=2),
    "Se": Atom(weight=78.971, covalence=2),
    "Si": Atom(weight=28.085, covalence=4),
}

DICT_EDGES = {
    "NoBond": 0,
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 1.5,
}

TYPE_EDGES = list(DICT_EDGES)
