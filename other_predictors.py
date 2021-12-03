import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("generated_data_1000.csv")
current = pd.get_dummies(data["Current state"], drop_first=True)
# print(current)
evidence = pd.get_dummies(data["Evidence"], drop_first=True)
# print(evidence)
X_train = pd.concat([current, evidence], axis=1)

y_train = data["Next state"]
# y_train = pd.get_dummies(data["Next state"])
# col = y_train.columns.tolist()
# new_col = []
# for i in col:
#     new_col.append("next_"+i)
# y_train.columns = new_col
# print(y_train)


test = pd.read_csv("balanced_test.csv")
cur_test = pd.get_dummies(test["Current state"], drop_first=True)
ev_test = pd.get_dummies(test["Evidence"], drop_first=True)
X_test = pd.concat([cur_test, ev_test], axis=1)

y_test = test["Next state"]
# y_test = pd.get_dummies(test["Next state"])
# y_test.columns = new_col
# print(y_test.iloc[6999].tolist())


# dtree = DecisionTreeClassifier()
# dtree.fit(X_train, y_train)
# predictions = dtree.predict(X_test)

# linear_reg = LogisticRegression()
# linear_reg.fit(X_train, y_train)
# predictions = linear_reg.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# error_rate = []
#
# # Will take some time
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
#          markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
