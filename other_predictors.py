import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt

data = pd.read_csv("generated_data_1000.csv")
current = pd.get_dummies(data["Current state"], drop_first=True)
# print(current)
evidence = pd.get_dummies(data["Evidence"], drop_first=True)
# print(evidence)
X_train = pd.concat([current, evidence], axis=1)

y_train = data["Next state"]

test = pd.read_csv("balanced_test_1000.csv")
cur_test = pd.get_dummies(test["Current state"], drop_first=True)
ev_test = pd.get_dummies(test["Evidence"], drop_first=True)
X_test = pd.concat([cur_test, ev_test], axis=1)

y_test = test["Next state"]

# log_reg = LogisticRegression(max_iter=200)
# log_reg.fit(X_train, y_train)
# predictions = log_reg.predict(X_test)

# naive_bayes = CategoricalNB()
# naive_bayes.fit(X_train, y_train)
# predictions = naive_bayes.predict(X_test)

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
# predictions = knn.predict(X_test)

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

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

print('DecisionTreeClassifier predictor')
print("Classification report")
print(classification_report(y_test, predictions))
print("Confusion matrix")
print(confusion_matrix(y_test, predictions))
