import plotly.graph_objects as go
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matchms import Spectrum

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


def _mol_to_data_uri(smiles: str, size: tuple[int, int] = (300, 300)) -> str | None:
    """Render a SMILES string to a base64 PNG data URI."""
    import base64
    from io import BytesIO

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def plot_proposals(
    query_spectrum: Spectrum,
    candidates: list,
    reference_row=None,
    mol_img_size: tuple[int, int] = (300, 300),
    output_html: str | None = None,
    true_smiles: str = None
):
    """
    Interactive visualization of structure proposals.

    - Top: interactive Bokeh mirror spectrum (query vs reference) with
      pan, zoom, hover tooltips showing m/z and intensity.
    - Bottom: grid of candidate molecule images with scores.

    Args:
        query_spectrum: The unknown query spectrum.
        candidates: List of StructureCandidate from infer_analog_structure.
        reference_row: DataFrame row (Series) with 'smiles', 'network_node_id',
                       'spectrum' fields for the reference compound. If None,
                       only the query spectrum is plotted.
        mol_img_size: Pixel size for each molecule image.
        output_html: If set, save a standalone HTML file at this path.
                     Otherwise the plot is shown in the browser / notebook.
        true_smiles: true smiles string for checking if available

    Returns:
        bokeh.layouts.column layout object.
    """
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import ColumnDataSource, HoverTool, Div
    from bokeh.layouts import column, row, grid

    # -----------------------------------------------------------------
    # 1. Interactive mirror spectrum
    # -----------------------------------------------------------------
    q_mz = np.asarray(query_spectrum.mz, dtype=np.float64)
    q_int = np.asarray(query_spectrum.intensities, dtype=np.float64)
    q_int_norm = q_int / q_int.max() if q_int.max() > 0 else q_int

    spec_fig = figure(
        width=900, height=400,
        x_axis_label="m/z",
        y_axis_label="Relative intensity",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="box_zoom",
    )
    spec_fig.y_range.start = -1.15
    spec_fig.y_range.end = 1.15

    # Query peaks (upward)
    q_src = ColumnDataSource(data=dict(
        mz=q_mz, intensity=q_int_norm,
        raw_intensity=q_int,
        zeros=np.zeros_like(q_mz),
    ))
    spec_fig.segment(x0="mz", y0="zeros", x1="mz", y1="intensity",
                     source=q_src, color="#1f77b4", line_width=1.5,
                     legend_label="Query")
    q_hover_glyphs = spec_fig.scatter(
        "mz", "intensity", source=q_src,
        size=5, color="#1f77b4", alpha=0.6,
    )

    title_parts = ["Query spectrum"]
    ref_div = None

    if reference_row is not None:
        ref_spec = reference_row["spectrum"]
        r_mz = np.asarray(ref_spec.mz, dtype=np.float64)
        r_int = np.asarray(ref_spec.intensities, dtype=np.float64)
        r_int_norm = r_int / r_int.max() if r_int.max() > 0 else r_int

        ref_name = reference_row.get("compound_name", reference_row.get("network_node_id", "Reference"))

        r_src = ColumnDataSource(data=dict(
            mz=r_mz, intensity=-r_int_norm,
            raw_intensity=r_int,
            zeros=np.zeros_like(r_mz),
        ))
        spec_fig.segment(x0="mz", y0="zeros", x1="mz", y1="intensity", source=r_src, color="#e07b54", line_width=1.5)
        r_hover_glyphs = spec_fig.scatter("mz", "intensity", source=r_src, size=5, color="#e07b54", alpha=0.6,
        )

        # Zero line
        spec_fig.line(
            [min(q_mz.min(), r_mz.min()) - 10, max(q_mz.max(), r_mz.max()) + 10],
            [0, 0], color="black", line_width=0.5,
        )

        title_parts = ["Query vs Reference", ref_name]

        # Hover for reference
        spec_fig.add_tools(HoverTool(
            renderers=[r_hover_glyphs],
            tooltips=[("m/z", "@mz{0.0000}"), ("intensity", "@raw_intensity{0.0}")],
        ))

        # Reference molecule image as inset
        ref_smiles = reference_row.get("smiles", "")
        if ref_smiles:
            uri = _mol_to_data_uri(ref_smiles, size=mol_img_size)
            if uri:
                img_w, img_h = mol_img_size
                ref_div = Div(text=(
                    '<div style="text-align:center;">'
                    '<h2 style="margin:0 0 4px 0;">Reference molecule</h2>'
                    f'<img src="{uri}" style="width:{img_w}px; height:{img_h}px; '
                    'border:1px solid #ccc; border-radius:4px;">'
                    '</div>'
                ), width=img_w + 40, height=img_h + 60)

        if true_smiles:
            true_uri = _mol_to_data_uri(true_smiles, size=mol_img_size)
            if uri:
                img_w, img_h = mol_img_size
                true_div = Div(text=(
                    '<div style="text-align:center;">'
                    '<h2 style="margin:0 0 4px 0;">True molecule</h2>'
                    f'<img src="{true_uri}" style="width:{img_w}px; height:{img_h}px; '
                    'border:1px solid #ccc; border-radius:4px;">'
                    '</div>'
                ), width=img_w + 40, height=img_h + 60)

    spec_fig.legend.location = "top_left"

    # Hover for query
    spec_fig.add_tools(HoverTool(
        renderers=[q_hover_glyphs],
        tooltips=[("m/z", "@mz{0.0000}"), ("intensity", "@raw_intensity{0.0}")],
    ))

    # Optional: reference molecule as a Div beside the title
    header_html = f"<h2 style='margin:0'>{title_parts[0]}</h2>"
    if len(title_parts) > 1:
        header_html = (
            f"<div style='display:flex; align-items:center;'>"
            f"<h2 style='margin:0'>{title_parts[0]}</h2></div>"
            f"<div>{title_parts[1]}</div>"
        )
    header_div = Div(text=header_html, width=900)

    # -----------------------------------------------------------------
    # 2. Molecule grid as HTML Divs
    # -----------------------------------------------------------------
    n_candidates = len(candidates)

    if n_candidates > 0:
        mol_divs = []
        for idx, candidate in enumerate(candidates):
            uri = _mol_to_data_uri(candidate.smiles, size=mol_img_size)
            img_tag = (
                f'<img src="{uri}" style="width:220px; height:220px;">'
                if uri else "<p>(invalid SMILES)</p>"
            )
            cell_html = (
                f'<div style="display:inline-block; text-align:center; '
                f'margin:8px; padding:8px; border:1px solid #ddd; '
                f'border-radius:6px; vertical-align:top; width:240px;">'
                f'<div style="font-weight:bold; font-size:13px; margin-bottom:4px;">'
                f'#{idx + 1}  {candidate.formula_delta}'
                f'</div>'
                f'{img_tag}'
                f'<div style="font-size:11px; margin-top:4px;">'
                f'frag={candidate.fragmentation_score:.2f} &nbsp; '
                f'loc={candidate.localization_score:.2f}'
                f'</div>'
                f'<div style="font-size:10px; color:gray; '
                f'word-break:break-all; margin-top:2px;">'
                f'{candidate.smiles}'
                f'</div>'
                f'</div>'
            )
            mol_divs.append(cell_html)

        grid_html = (
            '<div style="display:flex; flex-wrap:wrap; justify-content:center;">'
            + "".join(mol_divs)
            + '</div>'
        )
        mol_grid = Div(text=grid_html, width=900)
        layout = grid([header_div, [spec_fig, ref_div], [mol_grid, true_div]])


    else:
        layout = grid([header_div, [spec_fig, ref_div], [None, true_div]])

    # -----------------------------------------------------------------
    # 3. Output
    # -----------------------------------------------------------------
    if output_html:
        output_file(output_html, title="Structure Proposals")
        save(layout)
    else:
        show(layout)

    return layout
