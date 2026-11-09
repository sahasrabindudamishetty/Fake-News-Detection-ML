
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD TRAINED MODELS AND EVALUATION DATA
# ==========================================

print("Loading trained models and evaluation data...")

logistic_model = joblib.load(
    "models/logistic_regression_model.joblib"
)

naive_bayes_model = joblib.load(
    "models/naive_bayes_model.joblib"
)

evaluation_data = joblib.load(
    "models/evaluation_data.joblib"
)

X_train_tfidf = evaluation_data["X_train_tfidf"]
X_test_tfidf = evaluation_data["X_test_tfidf"]
y_train = evaluation_data["y_train"]
y_test = evaluation_data["y_test"]

print("All models and evaluation data loaded successfully!")


# ==========================================
# 2. GENERATE PREDICTIONS
# ==========================================

logistic_train_predictions = logistic_model.predict(X_train_tfidf)
logistic_test_predictions = logistic_model.predict(X_test_tfidf)

naive_bayes_train_predictions = naive_bayes_model.predict(X_train_tfidf)
naive_bayes_test_predictions = naive_bayes_model.predict(X_test_tfidf)

print("\nPredictions generated successfully!")


# ==========================================
# 3. EVALUATE MODEL PERFORMANCE
# ==========================================

def evaluate_model(name, y_true, y_pred):

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Fake News", "Real News"],
            zero_division=0
        )
    )

    return accuracy, precision, recall, f1


# ==========================================
# 4. TRAINING PERFORMANCE
# ==========================================

print("\n\n========== TRAINING PERFORMANCE ==========")

logistic_train_metrics = evaluate_model(
    "LOGISTIC REGRESSION - TRAINING",
    y_train,
    logistic_train_predictions
)

naive_bayes_train_metrics = evaluate_model(
    "NAIVE BAYES - TRAINING",
    y_train,
    naive_bayes_train_predictions
)


# ==========================================
# 5. TESTING PERFORMANCE
# ==========================================

print("\n\n========== TESTING PERFORMANCE ==========")

logistic_test_metrics = evaluate_model(
    "LOGISTIC REGRESSION - TESTING",
    y_test,
    logistic_test_predictions
)

naive_bayes_test_metrics = evaluate_model(
    "NAIVE BAYES - TESTING",
    y_test,
    naive_bayes_test_predictions
)


# ==========================================
# 6. CONFUSION MATRICES
# ==========================================

logistic_cm = confusion_matrix(
    y_test,
    logistic_test_predictions,
    labels=[0, 1]
)

naive_bayes_cm = confusion_matrix(
    y_test,
    naive_bayes_test_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    logistic_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake News", "Real News"],
    yticklabels=["Fake News", "Real News"]
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig("screenshots/logistic_confusion_matrix.png")
plt.close()


plt.figure(figsize=(7, 5))

sns.heatmap(
    naive_bayes_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Fake News", "Real News"],
    yticklabels=["Fake News", "Real News"]
)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig("screenshots/naive_bayes_confusion_matrix.png")
plt.close()

print("\nConfusion matrix graphs saved successfully!")


# ==========================================
# 7. TRAINING VS TESTING ACCURACY
# ==========================================

models = ["Logistic Regression", "Naive Bayes"]

training_accuracies = [
    logistic_train_metrics[0] * 100,
    naive_bayes_train_metrics[0] * 100
]

testing_accuracies = [
    logistic_test_metrics[0] * 100,
    naive_bayes_test_metrics[0] * 100
]

x = range(len(models))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    training_accuracies,
    width,
    label="Training Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    testing_accuracies,
    width,
    label="Testing Accuracy"
)

plt.xticks(list(x), models)
plt.ylabel("Accuracy (%)")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.ylim(0, 105)
plt.tight_layout()

plt.savefig("screenshots/training_testing_accuracy.png")
plt.close()


# ==========================================
# 8. MODEL PERFORMANCE COMPARISON
# ==========================================

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

logistic_values = [
    value * 100 for value in logistic_test_metrics
]

naive_bayes_values = [
    value * 100 for value in naive_bayes_test_metrics
]

x = range(len(metrics))

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    logistic_values,
    width,
    label="Logistic Regression"
)

plt.bar(
    [i + width / 2 for i in x],
    naive_bayes_values,
    width,
    label="Naive Bayes"
)

plt.xticks(list(x), metrics)
plt.ylabel("Score (%)")
plt.title("Model Performance Comparison")
plt.legend()
plt.ylim(0, 105)
plt.tight_layout()

plt.savefig("screenshots/model_performance_comparison.png")
plt.close()


# ==========================================
# 9. FINAL MODEL COMPARISON
# ==========================================

print("\n\n========== FINAL MODEL COMPARISON ==========")

print("\nLogistic Regression:")
print("Training Accuracy:", round(training_accuracies[0], 2), "%")
print("Testing Accuracy :", round(testing_accuracies[0], 2), "%")

print("\nNaive Bayes:")
print("Training Accuracy:", round(training_accuracies[1], 2), "%")
print("Testing Accuracy :", round(testing_accuracies[1], 2), "%")

print("\nBest Performing Model:")

if logistic_test_metrics[0] > naive_bayes_test_metrics[0]:
    print("Logistic Regression")
elif naive_bayes_test_metrics[0] > logistic_test_metrics[0]:
    print("Multinomial Naive Bayes")
else:
    print("Both models have equal testing accuracy.")


# ==========================================
# 10. COMPLETION
# ==========================================

print("\nAll evaluation graphs saved successfully!")

print("\nGenerated files:")
print("1. screenshots/logistic_confusion_matrix.png")
print("2. screenshots/naive_bayes_confusion_matrix.png")
print("3. screenshots/training_testing_accuracy.png")
print("4. screenshots/model_performance_comparison.png")

print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")
