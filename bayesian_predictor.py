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


print(df[parent][df[parent] == "S1"])
print(df[child][(df[parent] == "S1") & (df[child] == "S4")])


def calculate_prob(pd_frame, state1, state2) -> float:
    if len(pd_frame[parent][pd_frame[parent] == state1]) != 0:
        return len(pd_frame[child][(pd_frame[parent] == state1) & (pd_frame[child] == state2)]) / len(pd_frame[parent][pd_frame[parent] == state1])
    else:
        return 0.0


print(calculate_prob(df, "S1", "S4"))
