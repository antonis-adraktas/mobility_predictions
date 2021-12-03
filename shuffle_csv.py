import pandas as pd

df = pd.read_csv("generated_data_5000.csv")
sample = df.sample(frac=1, random_state=1)
states = sample["Next state"].unique().tolist()
# print(states)
new_dfs = []
for i in states:
    bal = sample[sample["Next state"] == i].iloc[:1000]
    new_dfs.append(bal)

balanced = pd.concat(new_dfs, axis=0)
balanced = balanced.sample(frac=1, random_state=1)
print(balanced.describe)
print(balanced.head)

balanced.to_csv("balanced_1000.csv", index=False)
