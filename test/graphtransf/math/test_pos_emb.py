from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.math.positional_emb import laplacian_embedding
from annotix_ml.graphtransf.test_utils import create_random_start

def test_laplacian_emb():
    n = 40
    bs = 64
    natoms = len(VALID_ELEMENTS)
    nbonds = len(TYPE_EDGES)
    k = 5

    N, E, mask = create_random_start(bs, n, nbonds, natoms)
    for edges, m in zip(E, mask):
        n_mol = int(m.sum(-1))
        adj = edges[:n_mol, :n_mol, :]
        eigvectors = laplacian_embedding(adj, k)

        assert eigvectors.shape == (n_mol, k), f"expected {(n_mol, k)} got {eigvectors.shape}"

#TODO : make test check that with predetermined adjacency matrix we have correct eigenvectors 


if __name__ == "__main__":
    test_laplacian_emb()
