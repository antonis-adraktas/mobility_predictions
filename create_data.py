import numpy as np
import pandas as pd


class GenerateData:
    column_names = ["Current state", "Next state"]
    states = ["S1", "S4", "S5", "S6", "S7", "S8", "S9"]
    evidence_dict = {
        "E1": "High rank employee",
        "E2": "Marketing employee",
        "E3": "Other employee",
        "E4": "Household work",
    }
    states_dict = {
        "S1": "Person is at home",
        "S4": "In office (cooperate)",
        "S5": "In head office",
        "S6": "Coming to office for lunch",
        "S7": "Attending some party or get together",
        "S8": "Leaving early from office for outside work",
        "S9": "In meeting outside office"
    }
    transition_probs = {
        "S1": [0, 0.8, 0.1, 0, 0.03, 0, 0.07],
        "S4": [0.3, 0, 0.2, 0.3, 0, 0.10, 0.10],
        "S5": [0.5, 0.2, 0, 0, 0.2, 0.1, 0],
        "S6": [0, 0.6, 0, 0, 0, 0.25, 0.15],
        "S7": [0.85, 0.05, 0.05, 0, 0, 0, 0.05],
        "S8": [0.9, 0, 0, 0, 0.1, 0, 0],
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

        self.add_evidence("S1", "S4", ["E1", "E2", "E3"], [0.05, 0.25, 0.7])
        self.add_evidence("S1", "S5", ["E1", "E2", "E3"], [0.65, 0.3, 0.05])
        self.add_evidence("S1", "S7", ["E1", "E2", "E3"], [0.4, 0.4, 0.2])
        self.add_evidence("S1", "S9", ["E1", "E2", "E3"], [0.35, 0.6, 0.05])
        self.add_evidence("S4", "S1", ["E1", "E2", "E3"], [0.03, 0.2, 0.77])
        self.add_evidence("S4", "S5", ["E1", "E2", "E3"], [0.3, 0.6, 0.1])
        self.add_evidence("S4", "S6", ["E1", "E2", "E3"], [0.05, 0.25, 0.7])
        self.add_evidence("S4", "S8", ["E4", "E2", "E1"], [0.7, 0.15, 0.15])
        self.add_evidence("S4", "S9", ["E2", "E1", "E3"], [0.6, 0.35, 0.05])
        self.add_evidence("S5", "S1", ["E1", "E2", "E3"], [0.7, 0.2, 0.1])
        self.add_evidence("S5", "S4", ["E1", "E2", "E3"], [0.3, 0.4, 0.3])
        self.add_evidence("S5", "S7", ["E1", "E2", "E3"], [0.4, 0.3, 0.3])
        self.add_evidence("S5", "S8", ["E4", "E1", "E2"], [0.6, 0.2, 0.2])
        self.add_evidence("S6", "S4", ["E3", "E1", "E2"], [0.8, 0.05, 0.15])
        self.add_evidence("S6", "S8", ["E4", "E1", "E2", "E3"], [0.65, 0.05, 0.15, 0.15])
        self.add_evidence("S6", "S9", ["E1", "E2", "E3"], [0.35, 0.55, 0.1])
        self.add_evidence("S7", "S1", ["E1", "E2", "E3"], [0.1, 0.2, 0.7])
        self.add_evidence("S7", "S4", ["E1", "E2", "E3"], [0.0, 0.4, 0.6])
        self.add_evidence("S7", "S5", ["E1", "E2"], [0.75, 0.25])
        self.add_evidence("S7", "S9", ["E1", "E2"], [0.6, 0.4])
        self.add_evidence("S8", "S1", ["E4", "E3", "E2", "E1"], [0.7, 0.22, 0.05, 0.03])
        self.add_evidence("S8", "S7", ["E3", "E2", "E1"], [0.6, 0.15, 0.25])
        self.add_evidence("S9", "S1", ["E3", "E2", "E1", "E4"], [0.3, 0.2, 0.1, 0.4])
        self.add_evidence("S9", "S4", ["E3", "E2", "E1"], [0.25, 0.6, 0.15])
        self.add_evidence("S9", "S5", ["E3", "E2", "E1"], [0.05, 0.3, 0.65])
        self.add_evidence("S9", "S7", ["E3", "E2", "E1"], [0.1, 0.4, 0.5])


# data = GenerateData(1000)
# data.fill_evidence()
# data.df.sample(frac=1, random_state=1).to_csv("test_data_1000.csv", index=False)
