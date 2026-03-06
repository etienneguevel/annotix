import plotly.graph_objects as go
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import gridspec
from matchms import Spectrum
from matchms.plotting import plot_spectra_mirror

def draw(network: nx.Graph, seed: int = 42) -> go.Figure:
    """
    Draw a molecular network as an interactive Plotly figure.

    - Hover over edges to see cosine weight and matched_peaks.
    - Hover over nodes to see node_id, compound_name, and SMILES.
    """
    pos = nx.spring_layout(network, seed=seed)

    # --- Edge lines ---
    edge_x, edge_y = [], []
    for u, v in network.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#aaa"),
        hoverinfo="none",
        showlegend=False,
    )

    # --- Invisible midpoint markers for edge hover ---
    mid_x, mid_y, edge_hover = [], [], []
    for u, v, data in network.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        weight = data.get("weight")
        matched_peaks = data.get("matched_peaks")
        weight_str = f"{weight:.3f}" if isinstance(weight, float) else str(weight)
        edge_hover.append(
            f"<b>{u} — {v}</b><br>"
            f"Cosine score: {weight_str}<br>"
            f"Matched peaks: {matched_peaks}"
        )

    edge_mid_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        hovertext=edge_hover,
        showlegend=False,
    )

    # --- Nodes: annotated vs unknown ---
    ann = {"x": [], "y": [], "hover": [], "label": []}
    unk = {"x": [], "y": [], "hover": [], "label": []}

    for node_id, data in network.nodes(data=True):
        x, y = pos[node_id]
        compound_name = data.get("compound_name") or "—"
        smiles = data.get("smiles") or "—"
        hover = (
            f"<b>{node_id}</b><br>"
            f"Compound: {compound_name}<br>"
            f"SMILES: {smiles}"
        )
        if data.get("compound_name"):
            ann["x"].append(x)
            ann["y"].append(y)
            ann["hover"].append(hover)
            ann["label"].append(compound_name)
        else:
            unk["x"].append(x)
            unk["y"].append(y)
            unk["hover"].append(hover)
            unk["label"].append("")

    annotated_trace = go.Scatter(
        x=ann["x"], y=ann["y"],
        mode="markers",
        marker=dict(size=14, color="#e07b54", line=dict(width=1.5, color="white")),
        textposition="top center",
        hoverinfo="text",
        hovertext=ann["hover"],
        name="Annotated",
    )

    unknown_trace = go.Scatter(
        x=unk["x"], y=unk["y"],
        mode="markers",
        marker=dict(size=14, color="#6baed6", line=dict(width=1.5, color="white")),
        hoverinfo="text",
        hovertext=unk["hover"],
        name="Unknown",
    )

    fig = go.Figure(
        data=[edge_trace, edge_mid_trace, annotated_trace, unknown_trace],
        layout=go.Layout(
            title="Molecular Network",
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20),
        ),
    )

    fig.show()
    return fig

def plot_molecule(smiles, network_node_id=""):
    mol = Chem.MolFromSmiles(smiles)
    fig = Draw.MolsToGridImage([mol], legends=[network_node_id], molsPerRow=1)
    return fig

def plot_spectrum(query_spectrum, ref_spectrum=None):
    """
    Plot one or two spectra.

    Args:
        query_spectrum (Spectrum): A query spectrum to plot
        ref_spectrum (Series, optional): Series object with smiles, network_node_id and spectrum fields
    """
    query_spectrum.set(key="compound_name", value="Query spectrum")
    if ref_spectrum is not None:
        img = plot_molecule(ref_spectrum["smiles"], network_node_id=ref_spectrum["network_node_id"])
        fig, ax = query_spectrum.plot_against(ref_spectrum["spectrum"], figsize=(4, 3))
        img = OffsetImage(img, zoom=.3)
        ab = AnnotationBbox(img, (350, -.5))
        ax.add_artist(ab)

    else:
        fig, ax = query_spectrum.plot()

    return fig, ax


def plot_proposals(
    query_spectrum: Spectrum,
    candidates: list,
    reference_row=None,
    mol_img_size: tuple[int, int] = (300, 300),
    cols: int = 3,
):
    """
    Plot all structure proposals in a single figure:
    - Top: mirror spectrum (query vs reference)
    - Bottom: grid of proposed candidate molecules with scores

    Args:
        query_spectrum: The unknown query spectrum.
        candidates: List of StructureCandidate from infer_analog_structure.
        reference_row: DataFrame row (Series) with 'smiles', 'network_node_id',
                       'spectrum' fields for the reference compound. If None,
                       only the query spectrum is plotted.
        mol_img_size: Pixel size for each molecule image.
        cols: Number of molecule columns in the grid.
    """
    n_candidates = len(candidates)
    if n_candidates == 0:
        return plot_spectrum(query_spectrum, reference_row)

    rows_mol = (n_candidates + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 3.5, 4 + rows_mol * 3))
    gs = gridspec.GridSpec(
        1 + rows_mol, cols,
        height_ratios=[1.5] + [1] * rows_mol,
        hspace=0.4, wspace=0.3,
    )

    # --- Top: mirror spectrum spanning full width ---
    ax_spec = fig.add_subplot(gs[0, :])
    query_spectrum.set(key="compound_name", value="Query spectrum")
    if reference_row is not None:
        ref_spec = reference_row["spectrum"]
        plot_spectra_mirror(query_spectrum, ref_spec, ax=ax_spec)
        ref_smiles = reference_row.get("smiles", "")
        ref_name = reference_row.get("compound_name", reference_row.get("network_node_id", ""))
        ax_spec.set_title(f"Query vs Reference: {ref_name}", fontsize=10)
        if ref_smiles:
            ref_mol = Chem.MolFromSmiles(ref_smiles)
            if ref_mol:
                ref_img = Draw.MolToImage(ref_mol, size=(200, 200))
                imagebox = OffsetImage(ref_img, zoom=0.4)
                ab = AnnotationBbox(imagebox, (0.85, -0.45), xycoords="axes fraction",
                                    frameon=True, bboxprops=dict(edgecolor="gray", alpha=0.8))
                ax_spec.add_artist(ab)
    else:
        query_spectrum.plot(ax=ax_spec)
        ax_spec.set_title("Query spectrum", fontsize=10)

    # --- Bottom: molecule grid ---
    for idx, candidate in enumerate(candidates):
        row_i = 1 + idx // cols
        col_i = idx % cols
        ax_mol = fig.add_subplot(gs[row_i, col_i])

        mol = Chem.MolFromSmiles(candidate.smiles)
        if mol:
            img = Draw.MolToImage(mol, size=mol_img_size)
            ax_mol.imshow(img)
        ax_mol.set_xticks([])
        ax_mol.set_yticks([])

        label = (
            f"#{idx + 1}  {candidate.formula_delta}\n"
            f"frag={candidate.fragmentation_score:.2f}  "
            f"loc={candidate.localization_score:.2f}"
        )
        ax_mol.set_title(label, fontsize=8, pad=4)
        ax_mol.set_xlabel(candidate.smiles, fontsize=6, color="gray")

    # Hide unused grid cells
    for idx in range(n_candidates, rows_mol * cols):
        row_i = 1 + idx // cols
        col_i = idx % cols
        ax_empty = fig.add_subplot(gs[row_i, col_i])
        ax_empty.set_visible(False)

    return fig
