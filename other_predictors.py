import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("generated_data_1000.csv")
# print(data.head)
current = pd.get_dummies(data["Current state"])
# print(current)
evidence = pd.get_dummies(data["Evidence"])
# print(evidence)
X_train = pd.concat([current, evidence], axis=1)
# print(X_train)

y_train = pd.get_dummies(data["Next state"])
col = y_train.columns.tolist()
new_col = []
for i in col:
    new_col.append("next_"+i)
y_train.columns = new_col
# print(y_train)


test = pd.read_csv("test_data_1000.csv")
cur_test = pd.get_dummies(test["Current state"])
ev_test = pd.get_dummies(test["Evidence"])
X_test = pd.concat([cur_test, ev_test], axis=1)
y_test = pd.get_dummies(test["Next state"])
y_test.columns = new_col
print(y_test.iloc[6999].tolist())


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(predictions[6999].tolist())
count = 0
for i in range(len(predictions)):
    if predictions[i].tolist() == y_test.iloc[i].tolist():
        # print(predictions[i].tolist(), y_test.iloc[i].tolist())
        count += 1

print("Accuracy ", count/len(predictions))

print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
