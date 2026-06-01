import pandas as pd

df = pd.read_csv("data/cleaned_jobs.csv")
df.to_parquet("data/cleaned_jobs.parquet", engine="pyarrow", index=False)
print("Done —", df.shape)