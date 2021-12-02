import pandas as pd
import numpy as np
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

data = pd.read_csv("generated_data_1000.csv")
df_columns = data.columns.tolist()
states = data[df_columns[0]].unique().tolist()


def get_state_paths(state) -> list:
    """Returns a sorted list of the possible next states from the current state"""
    return sorted(data[data[df_columns[0]] == state][df_columns[1]].unique().tolist())


def get_prob_transition_per_node(state) -> list:
    """Returns a list of probabilities transition based on the available paths for each state"""
    paths = get_state_paths(state)
    transition_freq = []
    for i in paths:
        transition_freq.append(data[(data[df_columns[0]] == state) & (data[df_columns[1]] == i)][df_columns[0]].count())
    return [float(x)/sum(transition_freq) for x in transition_freq]


bbn = Bbn()
nodes = []


def create_nodes():
    count = 0
    for i in states:
        nodes.append(BbnNode(Variable(count, i, get_state_paths(i)), get_prob_transition_per_node(i)))
        bbn.add_node(nodes[count])
        count += 1


def add_edges():
    for i in nodes:
        node_values = i.to_dict().get('variable').get('values')
        for j in nodes:
            if j.to_dict().get('variable').get('name') in node_values:
                bbn.add_edge(Edge(i, j, EdgeType.DIRECTED))


create_nodes()
add_edges()
print(bbn.edges.keys())
for i in nodes:
    print(i)


def display_graph():
    # Set node positions
    pos = {0: (-2, 2), 1: (-2, 0), 2: (-0.5, -2), 3: (0, 3), 4: (2, 2), 5: (1.5, 0), 6: (2, -2)}

    # Set options for graph looks
    options = {
        "font_size": 16,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "edge_color": "red",
        "linewidths": 4,
        "width": 5, }

    # Generate graph
    n, d = bbn.to_nx_graph()
    nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

    # Update margins and print the graph
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()


display_graph()
# # Convert the BBN to a join tree
# This Inference controller function gives an error "list index out of range".
# It requires all sample combinations to be present which by definition are not present in our case.
# join_tree = InferenceController.apply(bbn)