import argparse
import sys

from molecularnetwork import MolecularNetwork
import plotly.graph_objects as go
import networkx as nx
from loguru import logger
import pandas as pd
from rdkit import Chem

def test_molecular_network():
    """
    Test the MolecularNetwork class with a simple example.
    """
    # Define SMILES strings and classes
    smiles_list = ["CCO", "CCN", "CCC", "CCF"]
    classes = ["alcohol", "amine", "alkane", "fluoride"]

    # Create MolecularNetwork instance
    network = MolecularNetwork(descriptor="morgan2", sim_metric="tanimoto", sim_threshold=0.25)

    # Create graph from SMILES strings and classes
    graph = network.create_graph(smiles_list, classes)

    # Check the properties of the graph
    assert len(graph.nodes) == len(smiles_list)
    assert all(node["smiles"] in smiles_list for node in graph.nodes.values())
    assert all(node["categorical_label"] in classes for node in graph.nodes.values())
    
    logger.info("Test passed successfully!")

def print_graph_info(graph, node_names=None):
    """
    Print information about the graph.
    """
    logger.info(f"Number of nodes: {len(graph.nodes)}")
    logger.info(f"Number of edges: {len(graph.edges)}")
    logger.info("Node attributes:")
    for node in graph.nodes(data=True):
        logger.info(node)
    logger.info("Similarity matrix:")

    sim_list = []
    for idx_node1, idx_node2 in graph.edges():
        node1 = node_names[idx_node1] if node_names else idx_node1
        node2 = node_names[idx_node2] if node_names else idx_node2

        similarity = graph[idx_node1][idx_node2]["similarity"]
        sim_list.append((node1, node2, round(similarity, 2)))

    sim_df = pd.DataFrame(sim_list, columns=["Node1", "Node2", "Similarity"])
    print(pd.pivot_table(sim_df, index="Node1", columns="Node2", values="Similarity", fill_value=0))

def plot_graph(graph, labels=None):
    """
    Plot the graph using matplotlib.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=700, node_color="lightblue", font_size=10, font_color="black")
    plt.show()

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

def check_smiles(smile_string):
    """
    Check if the provided SMILES string is valid.
    """
    valid = True
    try:
        mol = Chem.MolFromSmiles(smile_string)
        valid = mol is not None
    except Exception:
        valid = False
    return valid

def arguments():
    parser = argparse.ArgumentParser(description="Molecular networking modelling")
    parser.add_argument("--test", type=str, default="unit", help="Which test to run")
    return parser.parse_args()

if __name__ == "__main__":

    args = arguments()

    if args.test == "unit":
        test_molecular_network()
        sys.exit(0)

    if args.test == "toy":
        logger.info("Toy example: running example with predefined SMILES and classes")

        smiles_list = ["CCO", "CCN", "CCC", "CCF", "NC(C)Cc1ccccc1", "CCC(C)CC(C)N"]
        classes = ["alcohol", "amine", "alkane", "fluoride", "Amphetamine", "Methylhexanamine"]

    if args.test == "real":
        logger.info("Real life example")

        dataset = pd.read_csv("main/test_data/test-data.csv", nrows=100)
        valid_smiles = dataset["smiles"].apply(check_smiles)
        logger.info(f"Number of valid SMILES: {valid_smiles.sum()} out of {len(dataset)}")

        if sum(valid_smiles) != len(dataset):
            dataset = dataset.loc[valid_smiles]
            print(len(dataset))

        smiles_list = dataset["smiles"].tolist()
        classes = dataset["compound_name"].tolist()

    network = MolecularNetwork(descriptor="morgan2", sim_metric="tanimoto", sim_threshold=0.85)
    #TODO how to add the MS/MS spectra to the network computation? Via another similarity metric, possibly a spec2vec model?
    # network.similarity_calculator = SimilarityWord2Vec(model, allowed_missing_percentage=5.0) need to be implemented
    # network.fingerprint_calculator = like FingerprintCalculator but from the peaks instead of smiles need to be implemented
    graph = network.create_graph(smiles_list, classes)
    print_graph_info(graph, node_names=smiles_list)


    # Plot using matplotlib
    # plot_graph(graph, labels={i: node["smiles"] for i, node in graph.nodes(data=True)})

    # Plot using plotly
    nx_graph = network.graph
    plot_graph_plotly(nx_graph, labels={i: node["smiles"] for i, node in graph.nodes(data=True)},
                      properties={"compound_name": [classes[node] for node in graph.nodes()]})

    # Communities in the graph
    communities = nx.community.louvain_communities(graph)
    logger.info(f"Number of communities found: {len(communities)}")
