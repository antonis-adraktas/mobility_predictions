import pandas as pd
from create_data import GenerateData
from sklearn.metrics import classification_report, confusion_matrix

# generate = GenerateData(1000)
# generate.fill_evidence()
data = pd.read_csv("generated_data_1000.csv")
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
        if sum_values != 0:
            norm_value = prob.get(i)/sum_values
            prob.update({i: norm_value})
        else:
            prob.update({i: 0})
    return prob


def fit(train) -> dict:
    transition_pred = {}
    for i in states:
        for j in evidence:
            transition_prob = probs(train, df_columns[1], states, df_columns[0], [i], df_columns[2], [j])
            if sum(transition_prob.values()) != 0:
                transition_pred.update({f"{i},{j}": list(transition_prob.keys())[list(transition_prob.values()).index(
                        max(transition_prob.values()))][12:14]})
            else:
                transition_pred.update({f"{i},{j}": None})
    return transition_pred


# trained_matrix = fit(data)


def predict(test: pd.DataFrame) -> list:
    trained_matrix = fit(data)
    columns = test.columns.tolist()
    predictions = []
    for i in range(len(test)):
        key = f"{test[columns[0]][i]},{test[columns[1]][i]}"
        predictions.append(trained_matrix.get(key))
    return predictions


test_df = pd.read_csv("test_data_1000.csv")
# print(test_df.head)
X_test = test_df.drop(df_columns[1], axis=1)
# print(X_test)
y_test = test_df[df_columns[1]]
# print(y_test.head)

pred = predict(X_test)
print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))