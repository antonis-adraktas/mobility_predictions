import pandas as pd

df = pd.read_csv("generated_data_5000.csv")
sample = df.sample(frac=0.5, random_state=1, ignore_index=True)
sample.to_csv("random_sample_2500.csv", index=False)
