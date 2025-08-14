from argparse import ArgumentParser

from loguru import logger
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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
    parser.add_argument("--model_path", type=str, default="./model.pt", help="Path to the trained model.")
    return parser.parse_args()


def compute_similarity(model, spec_1, spec_2, threshold):
    sim = spectrum_similarity(model, spec_1, spec_2)
    if sim < threshold:
        return 0
    return sim

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

Compound name: {properties.get("compound_name", [''])[node] if properties else ""}
""" for node in graph.nodes()]

    fig = go.Figure(data=[go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="black")),
                          go.Scatter(x=node_x, y=node_y, mode="markers", text=node_labels,
                                     marker=dict(size=20, color="lightblue"), textposition="top center")])
    
    fig.update_layout(showlegend=False)
    fig.show()

if __name__ == "__main__":

    args = arguments()
    
    logger.info("Loading model from {}", args.model_path)
    model = torch.load(args.model_path, map_location=torch.device("cpu"))

    logger.info("Load test data and get documents")
    test = pd.read_csv("main/test_data/test-data.csv")
    spectra = [Spectrum(row) for _, row in test.iterrows()]
    spectra = [s for s in spectra if len(s.mz) > 10] #TODO a hardcoded value for preprocessing the spectra in the training step, should be a parameter somewhere
    threshold = 0.7  # Define a threshold for similarity

    similarities = np.array([[compute_similarity(model, spec_1, spec_2, threshold) for spec_2 in spectra] for spec_1 in tqdm(spectra)])


    # plt.figure(figsize=(10, 8))
    # sns.heatmap(similarities, cmap="viridis")
    # plt.title("Spectrum Similarity Heatmap")
    # plt.xlabel("Spectrum Index")
    # plt.ylabel("Spectrum Index")
    # plt.tight_layout()
    # plt.show()

    # Create a graph from the similarity matrix
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

    # Draw the graph
    # plt.figure(figsize=(12, 10))
    # pos = nx.spring_layout(G, seed=42)
    # edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    # nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    # plt.title("Spectrum Similarity Network")
    # plt.show()

    plot_graph_plotly(G)
