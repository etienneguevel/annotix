from argparse import ArgumentParser

from loguru import logger
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from annotix_ml.spectrum import Spectrum
from annotix_ml.word2vec import spectrum_similarity
import networkx as nx
import plotly.graph_objects as go


def arguments():
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="Compute similarity between spectra.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model.pt",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold to create edges in the graph.",
    )
    return parser.parse_args()


def compute_similarity(model, spec_1, spec_2, threshold):
    sim = spectrum_similarity(model, spec_1, spec_2)
    return int(sim > threshold)


def plot_graph_plotly(graph, labels=None, properties=[]):
    """
    Plot the graph using plotly.
    """

    # Compute node positions using spring layout
    pos = nx.spring_layout(graph)
    # Assign positions to node attributes
    for node, position in pos.items():
        graph.nodes[node]["pos"] = position

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]["pos"]
        x1, y1 = graph.nodes[edge[1]]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    node_x = [graph.nodes[node]["pos"][0] for node in graph.nodes()]
    node_y = [graph.nodes[node]["pos"][1] for node in graph.nodes()]

    # Convert labels dict to list of labels in node order
    node_labels = labels
    if labels is not None and isinstance(labels, dict):
        node_labels = [
            f"""
Smiles: {labels[node]}

Compound name: {properties.get("compound_name", [""])[node] if properties else ""}
"""
            for node in graph.nodes()
        ]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="black")
            ),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                text=node_labels,
                marker=dict(size=20, color="lightblue"),
                textposition="top center",
            ),
        ]
    )

    fig.update_layout(showlegend=False)
    fig.show()


def similarity_to_graph(spectra, similarities):
    G = nx.Graph()

    # Add nodes for each spectrum
    for i in range(len(spectra)):
        G.add_node(i)

    # Add edges for similarities above the threshold
    for i in range(len(spectra)):
        for j in range(i + 1, len(spectra)):
            sim = similarities[i, j]
            if sim > 0:
                G.add_edge(i, j, weight=sim)
    return G


def load_data():
    test = pd.read_csv("main/test_data/test-data.csv")

    # Check for NaN in peaks_list and remove those rows
    test = test[~test["peaks_list"].isna()]

    spectra = [Spectrum(row) for _, row in test.iterrows()]
    spectra = [
        s for s in spectra if len(s.mz) > 10
    ]  # TODO a hardcoded value for preprocessing the spectra in the training step, should be a parameter somewhere

    references = pd.read_csv("main/test_data/references.csv")

    # Check for NaN in peaks_list and remove those rows
    references = references[references.charge == "1+"]
    references = references[~references["peaks_list"].isna()]

    references = references.sample(100, random_state=42)

    ref_spectra = [Spectrum(row) for _, row in references.iterrows()]
    ref_spectra = [
        s for s in ref_spectra if len(s.mz) > 10
    ]  # TODO a hardcoded value for preprocessing the spectra in the training step, should be a parameter somewhere

    for pos, s in enumerate(ref_spectra):
        try:
            s.document()
        except AttributeError:
            logger.warning(f"Spectrum {s.name} has no peaks, skipping")
            ref_spectra.pop(pos)

    return spectra, ref_spectra


if __name__ == "__main__":
    args = arguments()

    logger.info("Loading model from {}", args.model_path)
    model = torch.load(args.model_path, map_location=torch.device("cpu"))

    logger.info("Load test data and get documents")
    test_spectra, ref_spectra = load_data()

    logger.info("Compute similarities")
    documents = [s.document(processing_first=True) for s in ref_spectra]

    # Compute upper triangle only, avoid double computation
    n = len(documents)
    similarities = np.zeros((n, n), dtype=int)
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            sim = compute_similarity(model, documents[i], documents[j], args.threshold)
            similarities[i, j] = sim
            similarities[j, i] = sim  # Symmetric

    # Create a graph from the similarity matrix
    G = similarity_to_graph(ref_spectra, similarities)

    # Draw the graph
    plot_graph_plotly(G)

    try:
        communities = nx.community.louvain_communities(G)
        logger.success(f"Found {len(communities)} communities")
        print(communities)
        for i, community in enumerate(communities):
            logger.info(f"Community {i + 1}:")
            for node in community:
                spectrum = ref_spectra[node]
                logger.info(f" - {spectrum.name} (Smiles: {spectrum.smiles})")
    except Exception as e:
        logger.error(f"Error in community detection: {e}")
