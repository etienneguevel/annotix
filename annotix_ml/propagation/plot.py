import plotly.graph_objects as go
import networkx as nx


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
        marker=dict(size=9, color="#6baed6", line=dict(width=1, color="white")),
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
