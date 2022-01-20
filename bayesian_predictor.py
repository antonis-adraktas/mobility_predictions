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
            # Calculate probabilities
            for val in parent1bands:
                for val2 in childbands:
                    key = f"{child}: {val2}, {parent1}: {val}"
                    value = df[df[parent1] == val][child].tolist().count(val2)
                    prob.update({key: value})
        elif parent2 is not None:
            # Check if parent3 exists
            if parent3 is None:
                # Calculate probabilities
                for val in parent1bands:
                    for val2 in parent2bands:
                        for val3 in childbands:
                            key = f"{child}: {val3}, {parent1}: {val}, {parent2}: {val2}"
                            value = df[(df[parent1] == val) & (df[parent2] == val2)][child].tolist().count(val3)
                            prob.update({key: value})
            elif parent3 is not None:
                # Check if parent4 exists
                if parent4 is None:
                    # Calculate probabilities
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


def fit(train, child, parent1, parent2=None) -> dict:
    """This function provides a dictionary with the predicted state for all data combination
     based on a train dataset. The algorithm logic is to simply select the state with the
     maximum probability from possible states in the dataset. If parent 2 is set to None then
     the fit function will only account for current state column data"""
    transition_pred = {}
    parent1_list = train[parent1].unique().tolist()
    for i in parent1_list:
        if parent2 is not None:
            parent2_list = train[parent2].unique().tolist()
            for j in parent2_list:
                transition_prob = probs(train, child, parent1_list, parent1, [i], parent2, [j])
                if sum(transition_prob.values()) != 0:
                    transition_pred.update(
                        {f"{i},{j}": list(transition_prob.keys())[list(transition_prob.values()).index(
                            max(transition_prob.values()))][12:14]})
                else:
                    transition_pred.update({f"{i},{j}": None})
        else:
            # this branch calculates probs without evidence column data
            transition_prob = probs(train, df_columns[1], states, df_columns[0], [i])
            if sum(transition_prob.values()) != 0:
                transition_pred.update(
                    {f"{i}": list(transition_prob.keys())[list(transition_prob.values()).index(
                        max(transition_prob.values()))][12:14]})
            else:
                transition_pred.update({f"{i}": None})
    return transition_pred


# print(probs(data, df_columns[1], states, df_columns[0], ["S1"], df_columns[2], ["E1"]))


def predict(test: pd.DataFrame, fit_matrix: dict) -> list:
    """This function provides a list with the predicted next state for each
    row of the given dataset. It uses the fit function results, that is trained on
    a different dataset, to provide predictions """
    columns = test.columns.tolist()
    predictions = []
    for i in range(len(test)):
        if len(list(fit_matrix.keys())[0]) > 2:
            key = f"{test[columns[0]][i]},{test[columns[1]][i]}"
        else:
            # runs when evidence is not accounted
            key = f"{test[columns[0]][i]}"
        predictions.append(fit_matrix.get(key))
    return predictions


trained_matrix = fit(data, df_columns[1], df_columns[0], df_columns[2]) # with evidence
# trained_matrix = fit(data, df_columns[1], df_columns[0], None)   # without evidence

print(trained_matrix)
test_df = pd.read_csv("balanced_test_1000.csv")
# print(test_df.head)
X_test = test_df.drop(df_columns[1], axis=1)
# print(X_test)
y_test = test_df[df_columns[1]]

pred = predict(X_test, trained_matrix)
print('Bayesian probabilistic predictor')
print("Classification report")
print(classification_report(y_test, pred))
print("Confusion matrix")
print(confusion_matrix(y_test, pred))


# for i in evidence:
#     new_test_df = test_df[test_df["Evidence"] == i]
#     new_x_test = new_test_df.drop(df_columns[1], axis=1).reset_index(drop=True)
#     new_y_test = new_test_df[df_columns[1]]
#
#     predictions = predict(new_x_test, trained_matrix)
#
#     print('Bayesian probabilistic predictor for evidence ', i)
#     print("Classification report")
#     print(classification_report(new_y_test, predictions))
#     print("Confusion matrix")
#     print(confusion_matrix(new_y_test, predictions))


