import pandas as pd
import numpy as np
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
from create_data import GenerateData
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

data = GenerateData(1000)
data.fill_evidence()
df = data.df
df_columns = df.columns.tolist()
states = df[df_columns[0]].unique().tolist()
# print(states)


def get_state_paths(state) -> list:
    """Returns a sorted list of the possible next states from the current state"""
    return sorted(df[df[df_columns[0]] == state][df_columns[1]].unique().tolist())


def get_prob_transition_per_node(state) -> list:
    """Returns a list of probabilities transition based on the available paths for each state"""
    paths = get_state_paths(state)
    transition_freq = []
    for i in paths:
        transition_freq.append(df[(df[df_columns[0]] == state) & (df[df_columns[1]] == i)][df_columns[0]].count())
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


# display_graph()
# # Convert the BBN to a join tree
# join_tree = InferenceController.apply(bbn)


# Define a function for printing marginal probabilities
def probs(pd_frame, child, childbands, parent, parentbands):
    prob = []
    for val in parentbands:
        for val2 in childbands:
            prob.append(pd_frame[pd_frame[parent] == val][child].tolist().count(val2))
    prob = [i / sum(prob) for i in prob]
    return prob


prob = probs(df, df_columns[1], states, df_columns[0], ["S1"])
print(prob)



# Use the above function to print marginal probabilities
# print_probs()