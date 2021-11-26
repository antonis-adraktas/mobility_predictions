import pandas as pd
import numpy as np
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

print(df.head)
parent = data.column_names[0]
child = data.column_names[1]
evidence = data.column_names[2]


print(df[parent][df[parent] == "S1"])
print(df[child][(df[parent] == "S1") & (df[child] == "S4")])
print(df[child][(df[parent] == "S4") & (df[child] == "S5") & ((df[evidence] == "E1") | (df[evidence] == "E1/E2"))])


def get_unique_evidence():
    """This function retrieves the distinct evidence list (including None) from the imported dataframe"""
    unique_evidence = []
    evidence_list = df[evidence].unique().tolist()
    for i in evidence_list:
        if i is not None:
            for j in i.split("/"):
                if j not in unique_evidence:
                    unique_evidence.append(j)
        else:
            unique_evidence.append(None)
    return unique_evidence


def evidence_comb(values: list) -> list:
    """This function returns all values provided and their combinations in
    a value1/value2 format which is used in the evidence column of the dataframe"""
    unique_ev = get_unique_evidence()
    evidence_matrix = []
    for i in values:
        evidence_matrix.append(i)
        for j in unique_ev:
            if j != i:
                evidence_matrix.append(f"{i}/{j}")
    return evidence_matrix


# def calculate_prob(pd_frame, state1, state2, ev) -> float:
#     if ev is None:
#         if len(pd_frame[parent][pd_frame[parent] == state1]) != 0:
#             return len(pd_frame[child][(pd_frame[parent] == state1) & (pd_frame[child] == state2)]) \
#                    / len(pd_frame[parent][pd_frame[parent] == state1])
#         else:
#             return 0.0
#     else:
#         if len(pd_frame[parent][(pd_frame[parent] == state1) & (pd_frame[evidence].isin(evidence_comb(ev)))]) != 0:
#             return len(pd_frame[child][(pd_frame[parent] == state1) & (pd_frame[child] == state2) &
#                                        (pd_frame[evidence].isin(evidence_comb(ev)))]) \
#                    / len(pd_frame[parent][(pd_frame[parent] == state1) & (pd_frame[evidence].isin(evidence_comb(ev)))])
#         else:
#             return 0.0

# print(df[parent][(df[parent] == "S4") & (df[evidence].isin(evidence_comb(["E1"])))].groupby(df[child], dropna=False).count())

print(get_unique_evidence())
print(evidence_comb(["E1", "E2"]))

