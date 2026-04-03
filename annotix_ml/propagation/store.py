
"""Utilities to serialize/deserialize NetworkX graphs (nodes, edges, attrs).

These functions produce JSON-serializable structures and reconstruct
NetworkX graphs. Spectrum-like objects (matchms.Spectrum or similar)
are converted to simple dicts with `mz`, `intensities` and `metadata`.
"""

from copy import deepcopy
import networkx as nx
import numpy as _np
try:
    from matchms import Spectrum as MatchmsSpectrum
except Exception:
    MatchmsSpectrum = None


def _serialize_value(val):
    """Return a JSON-serializable representation for common types."""
    # Spectrum-like objects
    if MatchmsSpectrum is not None and isinstance(val, MatchmsSpectrum):
        return {
            "__type__": "spectrum",
            "mz": list(map(float, getattr(val, "mz", []))),
            "intensities": list(map(float, getattr(val, "intensities", []))),
            "metadata": dict(getattr(val, "metadata", {}) or {})
        }

    # numpy arrays
    if isinstance(val, (_np.ndarray,)):
        return val.tolist()

    # dicts -> recursively serialize
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}

    # lists/tuples -> recursively serialize
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]

    # primitives (int/float/str/bool/None)
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val

    # fallback: use str()
    return str(val)


def _deserialize_value(val):
    """Inverse of _serialize_value for supported types."""
    if isinstance(val, dict) and val.get("__type__") == "spectrum":
        mz = _np.array(val.get("mz", []), dtype=float)
        intensities = _np.array(val.get("intensities", []), dtype=float)
        metadata = dict(val.get("metadata", {}) or {})
        if MatchmsSpectrum is not None:
            try:
                return MatchmsSpectrum(mz=mz, intensities=intensities, metadata=metadata)
            except Exception:
                # fallback to simple dict if construction fails
                return {"mz": mz, "intensities": intensities, "metadata": metadata}
        return {"mz": mz, "intensities": intensities, "metadata": metadata}

    if isinstance(val, dict):
        return {k: _deserialize_value(v) for k, v in val.items()}

    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]

    return val


def graph_to_json(network):
    """Serialize a NetworkX graph to a JSON-friendly dict.

    The returned structure is compatible with `networkx.readwrite.json_graph`
    node-link format, with node/edge attributes converted to JSON-serializable
    types. Does not mutate the original graph.
    """
    H = network.__class__()
    # copy graph-level attributes
    H.graph.update(deepcopy(network.graph))

    # copy nodes with serialized attributes
    for n, data in network.nodes(data=True):
        attrs = {k: _serialize_value(v) for k, v in data.items()}
        H.add_node(n, **attrs)

    # copy edges with serialized attributes
    if network.is_multigraph():
        for u, v, key, data in network.edges(keys=True, data=True):
            attrs = {k: _serialize_value(vv) for k, vv in data.items()}
            H.add_edge(u, v, key=key, **attrs)
    else:
        for u, v, data in network.edges(data=True):
            attrs = {k: _serialize_value(vv) for k, vv in data.items()}
            H.add_edge(u, v, **attrs)

    return nx.readwrite.json_graph.node_link_data(H)


def json_to_graph(data):
    """Reconstruct a NetworkX graph from JSON produced by `graph_to_json`.

    This will attempt to reconstruct spectrum objects as `matchms.Spectrum`
    when possible; otherwise spectra are returned as dicts with numpy arrays.
    """
    G = nx.readwrite.json_graph.node_link_graph(deepcopy(data))

    # deserialize node attributes
    for n, data in G.nodes(data=True):
        for k, v in list(data.items()):
            data[k] = _deserialize_value(v)

    # deserialize edge attributes
    if G.is_multigraph():
        for u, v, key, data in G.edges(keys=True, data=True):
            for k, v in list(data.items()):
                data[k] = _deserialize_value(v)
    else:
        for u, v, data in G.edges(data=True):
            for k, v in list(data.items()):
                data[k] = _deserialize_value(v)

    return G


def test_graph_serialization_roundtrip():
    """Unit test: create graph with graph/node/edge attributes, serialize and
    deserialize and assert equality of structure and attributes.
    """
    # build graph with graph-level, node and edge attributes
    G = nx.Graph(name="testnet", version=1)
    G.graph["description"] = "roundtrip test"

    # node attributes include primitives and numpy arrays
    G.add_node("n1", node_id="n1", precursor_mz=100.0, smiles="C",
               arr=_np.array([1, 2, 3]), meta={"k": "v"})
    G.add_node("n2", node_id="n2", precursor_mz=200.0, smiles="CC",
               arr=_np.array([4, 5]), meta={"k": "w"})

    # edge with attribute
    G.add_edge("n1", "n2", weight=0.85, label="link")

    data = graph_to_json(G)
    G2 = json_to_graph(data)

    def _equal_vals(a, b):
        # matchms.Spectrum objects
        if MatchmsSpectrum is not None and isinstance(a, MatchmsSpectrum) and isinstance(b, MatchmsSpectrum):
            return (_np.allclose(getattr(a, "mz", []), getattr(b, "mz", [])) and
                    _np.allclose(getattr(a, "intensities", []), getattr(b, "intensities", [])) and
                    dict(getattr(a, "metadata", {})) == dict(getattr(b, "metadata", {})))

        if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
            try:
                return _np.array_equal(_np.asarray(a), _np.asarray(b))
            except Exception:
                return False

        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_equal_vals(a[k], b[k]) for k in a.keys())

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_equal_vals(x, y) for x, y in zip(a, b))

        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) <= 1e-8

        return a == b

    # compare graph-level attrs
    assert set(G.graph.keys()) == set(G2.graph.keys())
    assert _equal_vals(G.graph, G2.graph)

    # compare nodes
    assert set(G.nodes) == set(G2.nodes)
    for n in G.nodes:
        a = G.nodes[n]
        b = G2.nodes[n]
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            assert _equal_vals(a[k], b[k])

    # compare edges
    assert set(G.edges) == set(G2.edges)
    for u, v in G.edges:
        a = G.edges[u, v]
        b = G2.edges[u, v]
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            assert _equal_vals(a[k], b[k])
