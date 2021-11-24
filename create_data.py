import numpy as np
import pandas as pd

column_names = ["Current state", "Next state"]


def generate_data(num_samples: int) -> pd.DataFrame:

    """This function will generate synthetic data for  mobility prediction based on
    the states and probabilities presented on paper Efficient mobility prediction scheme for pervasive networks.
    num_samples defines the number of samples to be produced for each state.
    :type num_samples: int"""

    states = ["S1", "S4", "S5", "S6", "S7", "S8", "S9"]
    current_state=[]
    next_state = []
    transition_probs = {
        "S1": [0, 0.8, 0.1, 0.05, 0.05, 0, 0],
        "S8": [0.9, 0, 0, 0, 0.1, 0, 0],
        "S7": [0.8, 0.2, 0, 0, 0, 0, 0],
        "S4": [0, 0, 0.2, 0.5, 0, 0.15, 0.15],
        "S5": [0.5, 0.2, 0, 0, 0.2, 0.1, 0],
        "S6": [0, 0.6, 0, 0, 0, 0.25, 0.15],
        "S9": [0.55, 0.2, 0.2, 0, 0.05, 0, 0]
    }

    for i in states:
        for k in range(num_samples):
            current_state.append(i)
            if i in transition_probs.keys():
                next_state.append(np.random.choice(states, p=transition_probs[i]))
            else:
                next_state.append(np.nan)

    variables = [current_state, next_state]
    df = pd.DataFrame(variables).transpose()
    df.columns = column_names

    ev_column = "Evidence"
    df[ev_column] = None
    column_names.append(ev_column)

    def add_evidence(state1: str, state2: str, evidence: list, probability: list):
        """This function adds evidence data on the given transition from state1 to state2
        based on the evidence and probability lists provided"""
        for i in range(len(df)):
            if df[column_names[0]][i] == state1 and df[column_names[1]][i] == state2:
                df[ev_column][i] = np.random.choice(evidence, p=probability)

    add_evidence("S4", "S5", ["E1/E2", "E1", None], [0.4, 0.4, 0.2])
    add_evidence("S4", "S8", ["E3", None], [0.8, 0.2])
    add_evidence("S4", "S9", ["E2", None], [0.6, 0.4])

    return df


my_df = generate_data(1000)
print(my_df.head)
# my_df.to_csv("my_data.csv")
# print(type(my_df['Next state'][6999]))

unique_states = my_df["Current state"].unique()
for i in unique_states:
    print("Groupby state: " + i)
    print(my_df[my_df["Current state"] == i].groupby("Next state").count())
    print("\n")

print(column_names)
print("S4/S5 Evidence", my_df[(my_df[column_names[0]] == "S4") & (my_df[column_names[1]] == "S5")].groupby(column_names[2], dropna=False).count())
print("S4/S8 Evidence", my_df[(my_df[column_names[0]] == "S4") & (my_df[column_names[1]] == "S8")].groupby(column_names[2], dropna=False).count())
print("S4/S9 Evidence", my_df[(my_df[column_names[0]] == "S4") & (my_df[column_names[1]] == "S9")].groupby(column_names[2], dropna=False).count())


help(generate_data)
