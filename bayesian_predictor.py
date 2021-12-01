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

# generate = GenerateData(1000)
# generate.fill_evidence()
data = pd.read_csv("generated_data_5000.csv")
df_columns = data.columns.tolist()
states = data[df_columns[0]].unique().tolist()
evidence = data[df_columns[2]].unique().tolist()
# print(states)
# print(df_columns)


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


# display_graph()
# # Convert the BBN to a join tree
# This Inference controller function gives an error "list index out of range".
# It requires all sample combinations to be present which by definition are not present in our case.
# join_tree = InferenceController.apply(bbn)


# Define a function for printing marginal probabilities
def probs(df, child, childbands,
          parent1=None, parent1bands=None,
          parent2=None, parent2bands=None,
          parent3=None, parent3bands=None,
          parent4=None, ):
    """This function provides a dictionary with the marginal probabilities for a classification
     problem with up to 3 parents. The keys of the dictionary provide info on what the exact column
      value is in a {child}: {val4}, {parent1}: {val}, {parent2}: {val2}, {parent3}: {val3} format"""
    # Initialize empty list
    prob = {}
    if parent1 is None:
        # Calculate probabilities
        for val in childbands:
            key = f"{child}: {val}"
            value = df[child].tolist().count(val)
            prob.update({key: value})
    elif parent1 is not None:
        # Check if parent2 exists
        if parent2 is None:
            # Calcucate probabilities
            for val in parent1bands:
                for val2 in childbands:
                    key = f"{child}: {val2}, {parent1}: {val}"
                    value = df[df[parent1] == val][child].tolist().count(val2)
                    prob.update({key: value})
        elif parent2 is not None:
            # Check if parent3 exists
            if parent3 is None:
                # Calcucate probabilities
                for val in parent1bands:
                    for val2 in parent2bands:
                        for val3 in childbands:
                            key = f"{child}: {val3}, {parent1}: {val}, {parent2}: {val2}"
                            value = df[(df[parent1] == val) & (df[parent2] == val2)][child].tolist().count(val3)
                            prob.update({key: value})
            elif parent3 is not None:
                # Check if parent4 exists
                if parent4 is None:
                    # Calcucate probabilities
                    for val in parent1bands:
                        for val2 in parent2bands:
                            for val3 in parent3bands:
                                for val4 in childbands:
                                    key = f"{child}: {val4}, {parent1}: {val}, {parent2}: {val2}, {parent3}: {val3}"
                                    value = df[(df[parent1] == val) & (df[parent2] == val2) & (df[parent3] == val3)][child].tolist().count(val4)
                                    prob.update({key: value})
    sum_values = sum(prob.values())
    for i in prob.keys():
        norm_value = prob.get(i)/sum_values
        prob.update({i: norm_value})
    return prob


probability = probs(data, df_columns[1], states, df_columns[0], ["S4"], df_columns[2], ["E3"])
print(probability)
print(probability.values())
