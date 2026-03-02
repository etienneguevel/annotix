
def graph_to_json(network):
    """
    From nx graph to list of dict, to be serialized
    """
    data = list(network.nodes.data()).copy()
    data_list = []
    for (nodeid, content) in data:
        nodecontent = {
            "spectrum": content.get("spectrum").to_dict(),
            "node_id": content["node_id"],
            "precursor_mz": content["precursor_mz"],
            "compound_name": content["compound_name"],
            "smiles": content["smiles"],
            "inchi": content["inchi"],
            "molecular_formula": content["molecular_formula"],
            "adduct": content["adduct"]
        }
        data_list.append((nodeid, nodecontent))
    return data_list

def json_to_graph(data_list):
    pass