import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD FAKE AND REAL NEWS DATASETS
# ==========================================

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print("FAKE NEWS DATASET")
fake.info()

print("\nREAL NEWS DATASET")
real.info()


# ==========================================
# 2. CHECK AND REMOVE DUPLICATES
# ==========================================

print("\nFAKE NEWS DUPLICATES:")
print(fake.duplicated().sum())

print("\nREAL NEWS DUPLICATES:")
print(real.duplicated().sum())

fake = fake.drop_duplicates()
real = real.drop_duplicates()

print("\nDATASET SIZES AFTER DUPLICATE REMOVAL")
print("Fake news:", fake.shape)
print("Real news:", real.shape)


# ==========================================
# 3. ASSIGN LABELS
# ==========================================

fake["label"] = 0
real["label"] = 1


# ==========================================
# 4. COMBINE BOTH DATASETS
# ==========================================

data = pd.concat([fake, real], ignore_index=True)

print("\nCOMBINED DATASET")
print("Shape:", data.shape)

print("\nLABEL DISTRIBUTION")
print(data["label"].value_counts())

print("\nCLASS DISTRIBUTION PERCENTAGE")
print(data["label"].value_counts(normalize=True) * 100)


# ==========================================
# 5. COMBINE TITLE AND ARTICLE TEXT
# ==========================================

data["content"] = (
    data["title"].fillna("") + " " +
    data["text"].fillna("")
)

print("\nSAMPLE COMBINED NEWS CONTENT:")
print(data["content"].iloc[0])


# ==========================================
# 6. CHECK MISSING VALUES
# ==========================================

print("\nMISSING VALUES:")
print(data[["content", "label"]].isnull().sum())


# ==========================================
# 7. DISPLAY FEATURES AND TARGET
# ==========================================

X = data["content"]
y = data["label"]

print("\nFEATURES AND TARGET")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 8. VISUALIZE CLASS DISTRIBUTION
# ==========================================

data["label"].value_counts().sort_index().plot(
    kind="bar",
    figsize=(7, 5),
    color=["tomato", "seagreen"]
)

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Label (0 = Fake, 1 = Real)")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("screenshots/class_distribution.png")
plt.close()

print("\nClass distribution graph saved!")


# ==========================================
# 9. SAVE PROCESSED DATASET
# ==========================================

data.to_csv("dataset/processed_news.csv", index=False)

print("\nProcessed dataset saved successfully!")
print("File: dataset/processed_news.csv")