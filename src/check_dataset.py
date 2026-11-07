import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print("FAKE NEWS DATASET")
print("Shape:", fake.shape)
print("Columns:", fake.columns.tolist())

print("\nREAL NEWS DATASET")
print("Shape:", real.shape)
print("Columns:", real.columns.tolist())