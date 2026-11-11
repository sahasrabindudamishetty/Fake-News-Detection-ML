import joblib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
# Load the fake and real news datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv") 
print("FAKE NEWS DATASET")
print(fake.info())

print("\nREAL NEWS DATASET")
print(real.info())
# Check duplicate records
print("\nFAKE NEWS DUPLICATES:")
print(fake.duplicated().sum())

print("\nREAL NEWS DUPLICATES:")
print(real.duplicated().sum())
# Remove duplicate records
fake = fake.drop_duplicates()
real = real.drop_duplicates()

# Display dataset sizes after removing duplicates
print("\nDATASET SIZES AFTER DUPLICATE REMOVAL")
print("Fake news:", fake.shape)
print("Real news:", real.shape)
# Assign labels to datasets
fake["label"] = 0
real["label"] = 1
# Combine fake and real datasets
data = pd.concat([fake, real], ignore_index=True)
# Display combined dataset information
print("\nCOMBINED DATASET")
print(data.shape)

print("\nLABEL DISTRIBUTION")
print(data["label"].value_counts())

# Calculate class distribution percentage
print("\nCLASS DISTRIBUTION PERCENTAGE")
print(data["label"].value_counts(normalize=True) * 100)

# Combine title and article text
data["content"] = data["title"] + " " + data["text"]

# Display sample combined content
print("\nSAMPLE COMBINED NEWS CONTENT:")
print(data["content"].iloc[0])


# Check missing values in the combined dataset
print("\nMISSING VALUES:")
print(data[["content", "label"]].isnull().sum())

# Separate features and target
X = data["content"]
y = data["label"]

# Display feature and target information
print("\nFEATURES AND TARGET")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())

# Visualize class distribution
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
print("\nGRAPH WINDOW CLOSED - CONTINUING EXECUTION")


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset split information
print("\nDATASET SPLIT")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))



# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Learn vocabulary and transform training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data using the same vocabulary
X_test_tfidf = tfidf.transform(X_test)

# Display TF-IDF information
print("\nTF-IDF VECTORIZATION")
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)
print("Number of features:", len(tfidf.get_feature_names_out()))

# Initialize Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model using TF-IDF training data
logistic_model.fit(X_train_tfidf, y_train)

print("\nLOGISTIC REGRESSION MODEL")
print("Model training completed successfully!")


# Initialize Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()  
 #MultinomialNB() creates the Naive Bayes classifier.

# Train the model using TF-IDF training data
naive_bayes_model.fit(X_train_tfidf, y_train)
 
print("\nMULTINOMIAL NAIVE BAYES MODEL")
print("Model training completed successfully!")


# Generate predictions using Logistic Regression
y_pred_logistic = logistic_model.predict(X_test_tfidf)

# Generate predictions using Multinomial Naive Bayes
y_pred_nb = naive_bayes_model.predict(X_test_tfidf)

print("\nPREDICTIONS GENERATED")
print("Logistic Regression predictions:", len(y_pred_logistic))
print("Naive Bayes predictions:", len(y_pred_nb))


# Evaluate Logistic Regression
print("\n========== LOGISTIC REGRESSION EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Precision:", precision_score(y_test, y_pred_logistic, average="weighted"))
print("Recall:", recall_score(y_test, y_pred_logistic, average="weighted"))
print("F1-Score:", f1_score(y_test, y_pred_logistic, average="weighted"))


# Evaluate Multinomial Naive Bayes
print("\n========== NAIVE BAYES EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, average="weighted"))
print("Recall:", recall_score(y_test, y_pred_nb, average="weighted"))
print("F1-Score:", f1_score(y_test, y_pred_nb, average="weighted"))

# ==========================================
# CLASSIFICATION REPORTS
# ==========================================

print("\n========== LOGISTIC REGRESSION CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred_logistic,
    target_names=["Fake News", "Real News"]
))


print("\n========== NAIVE BAYES CLASSIFICATION REPORT ==========")

print(classification_report(
    y_test,
    y_pred_nb,
    target_names=["Fake News", "Real News"]
))

# Confusion Matrix - Logistic Regression
cm_logistic = confusion_matrix(y_test, y_pred_logistic)

print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print(cm_logistic)


# Confusion Matrix - Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)

print("\nNAIVE BAYES CONFUSION MATRIX")
print(cm_nb)

# Plot Logistic Regression Confusion Matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Fake News", "Real News"],
    yticklabels=["Fake News", "Real News"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("screenshots/logistic_confusion_matrix.png")
plt.close()

print("\nLogistic Regression confusion matrix graph saved!")


# Plot Naive Bayes Confusion Matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_nb,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake News", "Real News"],
    yticklabels=["Fake News", "Real News"]
)

plt.title("Multinomial Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig("screenshots/naive_bayes_confusion_matrix.png")
plt.close()

print("\nNaive Bayes confusion matrix graph saved!")


# TRAINING ACCURACY

y_train_pred_logistic = logistic_model.predict(X_train_tfidf)
y_train_pred_nb = naive_bayes_model.predict(X_train_tfidf)

train_accuracy_logistic = accuracy_score(y_train, y_train_pred_logistic)
train_accuracy_nb = accuracy_score(y_train, y_train_pred_nb)

print("\n========== TRAINING ACCURACY ==========")

print("Logistic Regression Training Accuracy:",
      train_accuracy_logistic)

print("Naive Bayes Training Accuracy:",
      train_accuracy_nb)

      # TRAINING VS TESTING ACCURACY COMPARISON

test_accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
test_accuracy_nb = accuracy_score(y_test, y_pred_nb)

models = ["Logistic Regression", "Naive Bayes"]

training_accuracy = [
    train_accuracy_logistic,
    train_accuracy_nb
]

testing_accuracy = [
    test_accuracy_logistic,
    test_accuracy_nb
]

x = range(len(models))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width/2 for i in x],
    training_accuracy,
    width,
    label="Training Accuracy"
)

plt.bar(
    [i + width/2 for i in x],
    testing_accuracy,
    width,
    label="Testing Accuracy"
)

plt.xticks(list(x), models)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy Comparison")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig("screenshots/training_testing_accuracy.png")
plt.close()

print("\nTraining vs Testing Accuracy graph saved!")

# ALGORITHM PERFORMANCE COMPARISON

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

logistic_scores = [
    accuracy_score(y_test, y_pred_logistic),
    precision_score(y_test, y_pred_logistic, average="weighted"),
    recall_score(y_test, y_pred_logistic, average="weighted"),
    f1_score(y_test, y_pred_logistic, average="weighted")
]

nb_scores = [
    accuracy_score(y_test, y_pred_nb),
    precision_score(y_test, y_pred_nb, average="weighted"),
    recall_score(y_test, y_pred_nb, average="weighted"),
    f1_score(y_test, y_pred_nb, average="weighted")
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width/2 for i in x],
    logistic_scores,
    width,
    label="Logistic Regression"
)

plt.bar(
    [i + width/2 for i in x],
    nb_scores,
    width,
    label="Naive Bayes"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score")
plt.title("Algorithm Performance Comparison")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()
plt.savefig("screenshots/model_performance_comparison.png")
plt.close()

print("\nAlgorithm performance comparison graph saved!")

# SAVE LOGISTIC REGRESSION MODEL

joblib.dump(
    logistic_model,
    "models/logistic_regression_model.joblib"
)

print("\nLogistic Regression model saved successfully!")

# SAVE NAIVE BAYES MODEL

joblib.dump(
    naive_bayes_model,
    "models/naive_bayes_model.joblib"
)

print("\nNaive Bayes model saved successfully!")

# SAVE TF-IDF VECTORIZER

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.joblib"
)

print("\nTF-IDF vectorizer saved successfully!")