from argparse import ArgumentParser

from pipeline import read_adjacency

def get_arguments():
    """
    Parse command-line arguments for source databases and output options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(prog="Annotix")
    parser.add_argument("--path", type=str, default="net.adjlist")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_arguments()
    network = read_adjacency(args.path)
    print(network.nodes.data())
    print(len(network.nodes), "nodes")
    print(len(network.edges), "edges")

    # for (id, attributes) in network.nodes.data():
    #     print(attributes)
    #     spec = attributes.get("spectrum")
    #     print(spec.intensities)
    #     print(spec.mz)