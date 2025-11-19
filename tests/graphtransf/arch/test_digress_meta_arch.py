import torch
from omegaconf import OmegaConf

from annotix_ml import ROOT
from annotix_ml.graphtransf.arch.digress_meta_arch import DigressMetaArch
from annotix_ml.graphtransf.data.atoms_data import VALID_ELEMENTS, TYPE_EDGES
from annotix_ml.graphtransf.test_utils import create_random_start


def test_digress_meta_arch_initialization():
    """Test basic initialization of DigressMetaArch"""
    d = 128
    de = 64
    dy = 32
    n_heads = 4
    n_layers = 2
    nodes_distribution = [0.1] * len(VALID_ELEMENTS)
    edges_distribution = [0.2] * len(TYPE_EDGES)
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
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        k=k,
        extra_features=[],
    )

    assert meta_arch.loss_ratio == loss_ratio
    assert meta_arch.device == device
    assert meta_arch.diffuser is not None
    assert meta_arch.noiser is not None
    assert meta_arch.loss is not None


def test_digress_meta_arch_initialization_from_config():
    # Load the arguments from the config file
    cfg = OmegaConf.load(ROOT / "graphtransf" / "configs" / "default_config.yaml")

    # init the model from the arguments within
    device = torch.device("cpu")
    nodes_distribution = [0.1] * len(VALID_ELEMENTS)
    edges_distribution = [0.2] * len(TYPE_EDGES)

    meta_arch = DigressMetaArch(
        d=cfg.model.d,
        de=cfg.model.de,
        dy=cfg.model.dy,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=cfg.model.diffusion_steps,
        loss_ratio=cfg.train.loss_ratio,
        device=device,
        k=cfg.model.num_ev,
        extra_features=cfg.model.extra_features,
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
    nodes_distribution = [0.1] * len(VALID_ELEMENTS)
    edges_distribution = [0.2] * len(TYPE_EDGES)
    diffusion_steps = 10
    loss_ratio = 0.5
    device = torch.device("cpu")

    meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        k=k,
        extra_features=["laplacian_embedding"],
    )

    # Create random edges and mask
    _, edges, mask = create_random_start(bs, n, len(TYPE_EDGES), len(VALID_ELEMENTS))
    pos_emb, y = meta_arch.compute_extra_features(edges, mask)

    print(pos_emb.shape)
    assert pos_emb.shape[-1] == bs
    assert y.shape[0] == bs


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
    nodes_distribution = torch.tensor([0.1] * len(VALID_ELEMENTS))
    edges_distribution = torch.tensor([0.2] * len(TYPE_EDGES))
    diffusion_steps = 10
    loss_ratio = 0.5
    device = torch.device("cpu")

    meta_arch = DigressMetaArch(
        d=d,
        de=de,
        dy=dy,
        n_heads=n_heads,
        n_layers=n_layers,
        nodes_distribution=nodes_distribution,
        edges_distribution=edges_distribution,
        diffusion_steps=diffusion_steps,
        loss_ratio=loss_ratio,
        device=device,
        k=k,
        extra_features=["laplacian_embedding", "node_cycle"],
    )

    # Create random batch
    N, E, mask = create_random_start(bs, n, len(TYPE_EDGES), len(VALID_ELEMENTS))
    batch = (N, E, mask)

    # Test forward_backward
    total_loss = meta_arch.forward_backward(batch)

    assert total_loss is not None
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0
