import pandas as pd


df = pd.read_excel("data/raw/hackathon_preli_dataset.xlsx")


print(df.head(10))


print("Dataset shape:", df.shape) 