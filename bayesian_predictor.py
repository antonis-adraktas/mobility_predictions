from create_data import GenerateData

data = GenerateData(5000)
data.fill_evidence()
df = data.df

print(df.head)

column_names = data.column_names
unique_states = df["Current state"].unique()
for i in unique_states:
    print("Groupby state: " + i)
    print(df[df["Current state"] == i].groupby("Next state").count())
    print("\n")

print(column_names)
print("S4/S5 Evidence", df[(df[column_names[0]] == "S4")
                                & (df[column_names[1]] == "S5")].groupby(column_names[2], dropna=False).count())
print("S4/S8 Evidence", df[(df[column_names[0]] == "S4")
                                & (df[column_names[1]] == "S8")].groupby(column_names[2], dropna=False).count())
print("S4/S9 Evidence", df[(df[column_names[0]] == "S4")
                                & (df[column_names[1]] == "S9")].groupby(column_names[2], dropna=False).count())

