import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# 1. LOAD PROCESSED DATASET
# ==========================================

data = pd.read_csv("dataset/processed_news.csv")

print("Processed dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Separate features and target
X = data["content"]
y = data["label"]

print("\nFEATURES AND TARGET")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ==========================================
# 2. SPLIT DATASET INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDATASET SPLIT")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 3. TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Learn vocabulary and transform training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF VECTORIZATION")
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)
print("Number of features:", len(tfidf.get_feature_names_out()))

# ==========================================
# 4. TRAIN LOGISTIC REGRESSION MODEL
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_tfidf, y_train)

print("\nLOGISTIC REGRESSION MODEL")
print("Model training completed successfully!")
# ==========================================
# 5. TRAIN MULTINOMIAL NAIVE BAYES MODEL
# ==========================================

naive_bayes_model = MultinomialNB()

naive_bayes_model.fit(X_train_tfidf, y_train)

print("\nMULTINOMIAL NAIVE BAYES MODEL")
print("Model training completed successfully!")


# ==========================================
# 6. SAVE TRAINED MODELS AND VECTORIZER
# ==========================================

joblib.dump(
    logistic_model,
    "models/logistic_regression_model.joblib"
)

joblib.dump(
    naive_bayes_model,
    "models/naive_bayes_model.joblib"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.joblib"
)

print("\nMODELS SAVED SUCCESSFULLY!")
print("Logistic Regression model saved.")
print("Naive Bayes model saved.")
print("TF-IDF vectorizer saved.")

# ==========================================
# 7. SAVE DATA FOR MODEL EVALUATION
# ==========================================

evaluation_data = {
    "X_train_tfidf": X_train_tfidf,
    "X_test_tfidf": X_test_tfidf,
    "y_train": y_train,
    "y_test": y_test
}

joblib.dump(
    evaluation_data,
    "models/evaluation_data.joblib"
)

print("\nEVALUATION DATA SAVED SUCCESSFULLY!")