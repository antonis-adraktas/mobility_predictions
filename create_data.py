import numpy as np
import pandas as pd


class GenerateData:
    column_names = ["Current state", "Next state"]
    states = ["S1", "S4", "S5", "S6", "S7", "S8", "S9"]
    transition_probs = {
        "S1": [0, 0.8, 0.1, 0, 0.03, 0, 0.07],
        "S8": [0.9, 0, 0, 0, 0.1, 0, 0],
        "S7": [0.8, 0.1, 0.1, 0, 0, 0, 0],
        "S4": [0, 0, 0.2, 0.5, 0, 0.15, 0.15],
        "S5": [0.5, 0.2, 0, 0, 0.2, 0.1, 0],
        "S6": [0, 0.6, 0, 0, 0, 0.25, 0.15],
        "S9": [0.55, 0.2, 0.2, 0, 0.05, 0, 0]
    }

    def __init__(self, num_samples: int):
        """This function will initialize a GenerateDara object, which contains a pandas Dataframe with the data created.
        This is synthetic data for  mobility prediction based on the states and probabilities presented on paper
        Efficient mobility prediction scheme for pervasive networks.
        num_samples defines the number of samples to be produced for each state.
        """
        self.num_samples = num_samples

        current_state = []
        next_state = []

        for i in self.states:
            for k in range(num_samples):
                current_state.append(i)
                if i in self.transition_probs.keys():
                    next_state.append(np.random.choice(self.states, p=self.transition_probs[i]))
                else:
                    next_state.append(np.nan)

        variables = [current_state, next_state]
        self.df = pd.DataFrame(variables).transpose()
        self.df.columns = self.column_names

        self.column_names.append("Evidence")
        self.df[self.column_names[2]] = None

    def add_evidence(self, state1: str, state2: str, evidence: list, probability: list):
        """This function adds evidence data on the given transition from state1 to state2
        based on the evidence and probability lists provided"""
        for i in range(len(self.df)):
            if self.df[self.column_names[0]][i] == state1 and self.df[self.column_names[1]][i] == state2:
                self.df[self.column_names[2]][i] = np.random.choice(evidence, p=probability)

    def fill_evidence(self):
        """This function fills the evidence column with synthetic data"""
        # In case an event is satisfied with either or multiple evidence format
        # evidence1/evidence2/evidence3 etc should be used
        #TODO Needs refactoring, Eevery line should have evidence
        self.add_evidence("S4", "S5", ["E1/E2", "E1", None], [0.4, 0.4, 0.2])
        self.add_evidence("S4", "S8", ["E3", None], [0.8, 0.2])
        self.add_evidence("S4", "S9", ["E2", None], [0.6, 0.4])
        self.add_evidence("S5", "S8", ["E3", "E1/E2", None], [0.8, 0.1, 0.1])
        self.add_evidence("S1", "S5", ["E1/E2", "E1", None], [0.5, 0.2, 0.3])
        self.add_evidence("S1", "S9", ["E2", None], [0.4, 0.6])
        self.add_evidence("S9", "S4", ["E2", None], [0.5, 0.5])
        self.add_evidence("S9", "S5", ["E1", None], [0.4, 0.6])
        self.add_evidence("S6", "S9", ["E2", "E1", None], [0.35, 0.2, 0.45])
        self.add_evidence("S9", "S7", ["E1", None], [0.6, 0.4])
        self.add_evidence("S5", "S8", ["E3", None], [0.7, 0.3])
        self.add_evidence("S7", "S5", ["E1", None], [0.65, 0.35])
        self.add_evidence("S8", "S7", ["E1/E2", None], [0.65, 0.35])
        self.add_evidence("S5", "S7", ["E1", None], [0.55, 0.45])
        self.add_evidence("S5", "S4", ["E2", None], [0.45, 0.55])


# data = GenerateData(1000)
# data.fill_evidence()
# print(data.df.head)
# print(data.df.dropna(axis=0))
#
#
# column_names = data.column_names
# unique_states = data.df["Current state"].unique()
# for i in unique_states:
#     print("Groupby state: " + i)
#     print(data.df[data.df["Current state"] == i].groupby("Next state").count())
#     print("\n")
#
# print(column_names)
# print("S4/S5 Evidence", data.df[(data.df[column_names[0]] == "S4")
#                                 & (data.df[column_names[1]] == "S5")].groupby(column_names[2], dropna=False).count())
# print("S4/S8 Evidence", data.df[(data.df[column_names[0]] == "S4")
#                                 & (data.df[column_names[1]] == "S8")].groupby(column_names[2], dropna=False).count())
# print("S4/S9 Evidence", data.df[(data.df[column_names[0]] == "S4")
#                                 & (data.df[column_names[1]] == "S9")].groupby(column_names[2], dropna=False).count())

