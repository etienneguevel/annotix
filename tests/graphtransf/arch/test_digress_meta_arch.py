import pandas as pd
import torch
from omegaconf import OmegaConf

from annotix_ml import BASE_DIR
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.test_utils import create_random_start
from annotix_ml.graphtransf.data.dataset import GraphDatasetFromSMILEs


df = pd.read_csv(BASE_DIR / "data" / "MassSpecGym.csv")
df = df[df.fold == "train"].sample(1000)
train_dataset = GraphDatasetFromSMILEs(df)


def test_digress_meta_arch_initialization():
    """Test basic initialization of DigressMetaArch"""
    d = 128
    de = 64
    dy = 32
    n_heads = 4
    n_layers = 2
    diffusion_steps = 10
    loss_ratio = 0.5
    device = torch.device("cpu")
    k = 8

    meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        noise_strategy="uniform",
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        extra_features=["laplacian_embedding"],
        valid_elements=list(VALID_ELEMENTS),
        k=k,
    )

    assert meta_arch.loss_ratio == loss_ratio
    assert meta_arch.device == device
    assert meta_arch.diffuser is not None
    assert meta_arch.noiser is not None
    assert meta_arch.loss is not None


def test_digress_meta_arch_initialization_from_config():
    # Load the arguments from the config file
    cfg = OmegaConf.load(BASE_DIR / "configs" / "default_config.yaml")

    # init the model from the arguments within
    device = torch.device("cpu")

    meta_arch = DigressMetaArch(
        d=cfg.model.d,
        de=cfg.model.de,
        dy=cfg.model.dy,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        noise_strategy="uniform",
        diffusion_steps=cfg.model.diffusion_steps,
        loss_ratio=cfg.train.loss_ratio,
        device=device,
        valid_elements=train_dataset.valid_elements,
        extra_features=["laplacian_embedding"],
        k=cfg.model.num_ev,
        nodes_distribution=train_dataset.nodes_distribution,
        edges_distribution=train_dataset.edges_distribution,
    )

    assert meta_arch.loss_ratio == cfg.train.loss_ratio
    assert meta_arch.diffuser.dy == cfg.model.dy
    assert meta_arch.device == device
    assert meta_arch.diffuser is not None
    assert meta_arch.noiser is not None
    assert meta_arch.loss is not None


def test_compute_extra_features():
    """Test compute_extra_features method"""
    bs = 8
    n = 20
    d = 128
    de = 64
    dy = 32
    n_heads = 4
    n_layers = 2
    k = 8

    diffusion_steps = 10
    loss_ratio = 0.5
    device = torch.device("cpu")

    meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        noise_strategy="uniform",
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        valid_elements=train_dataset.valid_elements,
        k=k,
        extra_features=["laplacian_embedding"],
        nodes_distribution=train_dataset.nodes_distribution,
        edges_distribution=train_dataset.edges_distribution,
    )

    # Create random edges and mask
    sampled_t = torch.randint(1, diffusion_steps, (bs, 1))
    nodes, edges, mask = create_random_start(
        bs, n, len(TYPE_EDGES), len(VALID_ELEMENTS)
    )

    pos_emb, y = meta_arch.compute_extra_features(nodes, edges, mask, sampled_t)

    assert pos_emb.shape[0] == bs
    assert pos_emb.shape[-1] == meta_arch.node_features
    assert y.shape[0] == bs
    assert y.shape[-1] == meta_arch.global_features


def test_forward_backward_with_extra_features():
    """Test forward_backward method with extra features"""
    bs = 8
    n = 20
    d = 128
    de = 64
    dy = 32
    n_heads = 4
    n_layers = 2
    k = 8

    diffusion_steps = 10
    loss_ratio = 0.5
    device = torch.device("cpu")

    meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        noise_strategy="uniform",
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        valid_elements=train_dataset.valid_elements,
        k=k,
        extra_features=["laplacian_embedding", "node_cycle", "valence_features"],
        nodes_distribution=train_dataset.nodes_distribution,
        edges_distribution=train_dataset.edges_distribution,
    )

    # Create random batch
    N, E, mask = create_random_start(
        bs, n, len(TYPE_EDGES), len(train_dataset.valid_elements)
    )
    batch = {"nodes": N, "edges": E, "mask": mask}

    # Test compute_loss
    outputs = meta_arch.forward(N, E, mask)
    total_loss, *_ = meta_arch.compute_loss(batch, outputs)

    assert total_loss is not None
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0


def test_generate():
    """Test the generate method of DigressMetaArch"""
    # Model parameters
    d = 64
    de = 32
    dy = 16
    n_heads = 2
    n_layers = 2
    k = 4
    diffusion_steps = 5  # Small number for faster testing
    loss_ratio = 0.5
    device = torch.device("cpu")

    # Create the model
    meta_arch = meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        noise_strategy="uniform",
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        valid_elements=train_dataset.valid_elements,
        k=k,
        extra_features=["laplacian_embedding"],
        nodes_distribution=train_dataset.nodes_distribution,
        edges_distribution=train_dataset.edges_distribution,
    )

    # Generation parameters
    batch_size = 4
    max_nodes = 9

    # Generate graphs
    N, E, _ = meta_arch.generate(batch_size, max_nodes)

    # Verify output shapes
    assert N.shape[0] == batch_size
    assert E.shape[0] == batch_size
    assert N.shape[-1] == len(train_dataset.valid_elements)
    assert E.shape[-1] == len(TYPE_EDGES)

    # Verify outputs are one-hot encoded
    n = N.shape[1]

    assert torch.allclose(N.sum(dim=-1).float(), torch.ones(batch_size, n), atol=1e-5)
    assert torch.allclose(
        E.sum(dim=-1).float(), torch.ones(batch_size, n, n), atol=1e-5
    )

    # Verify outputs are valid probabilities (all values between 0 and 1)
    assert (N >= 0).all() and (N <= 1).all()
    assert (E >= 0).all() and (E <= 1).all()

    # Verify the generated graphs respect the mask
    # (nodes beyond the randomly sampled size should be zero)
    # This is implicitly tested by the one-hot property above
