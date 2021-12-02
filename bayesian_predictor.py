import pandas as pd
from create_data import GenerateData
import time
from sklearn.metrics import classification_report, confusion_matrix

# generate = GenerateData(1000)
# generate.fill_evidence()
start = time.time_ns()
data = pd.read_csv("generated_data_1000.csv")
read_csv_time = time.time_ns()
df_columns = data.columns.tolist()
states = data[df_columns[0]].unique().tolist()
evidence = data[df_columns[2]].unique().tolist()


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


end = time.time_ns()

print("Time to read csv= ", (read_csv_time-start)/1000000000)
print("Time to process= ", (end - read_csv_time)/1000000000)


def predict(df):
    predictions = []
    for i in range(len(df)):
        trasition_prob = probs(df, df_columns[1], states, df_columns[0],
                               [df[df_columns[0]][i]], df_columns[2], [df[df_columns[2]][i]])
        prediction = list(trasition_prob.keys())[list(trasition_prob.values()).index(
            max(trasition_prob.values()))][12:14]
        predictions.append(prediction)
    return predictions


pred = predict(data)
print(classification_report(data[df_columns[1]], pred))

print(confusion_matrix(data[df_columns[1]], pred))
