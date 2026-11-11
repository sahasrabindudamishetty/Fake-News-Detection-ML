
import joblib

# ==========================================
# 1. LOAD TRAINED MODELS AND TF-IDF VECTORIZER
# ==========================================

logistic_model = joblib.load("models/logistic_regression_model.joblib")
naive_bayes_model = joblib.load("models/naive_bayes_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

print("Models and TF-IDF vectorizer loaded successfully!")


# ==========================================
# 2. GET NEWS ARTICLE FROM USER
# ==========================================

news = input("\nEnter the news article: ")

if not news.strip():
    print("Please enter a valid news article.")
    exit()

# Convert article into TF-IDF features
news_tfidf = tfidf.transform([news])

print("\nNews article converted into numerical features successfully!")


# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================

logistic_prediction = logistic_model.predict(news_tfidf)[0]
naive_bayes_prediction = naive_bayes_model.predict(news_tfidf)[0]


# ==========================================
# 4. CALCULATE PREDICTION PROBABILITIES
# ==========================================

logistic_probabilities = logistic_model.predict_proba(news_tfidf)[0]
naive_bayes_probabilities = naive_bayes_model.predict_proba(news_tfidf)[0]

logistic_confidence = logistic_probabilities[logistic_prediction] * 100
naive_bayes_confidence = naive_bayes_probabilities[naive_bayes_prediction] * 100


# ==========================================
# 5. CONVERT NUMERICAL PREDICTIONS INTO LABELS
# ==========================================

logistic_result = "Fake News" if logistic_prediction == 0 else "Real News"
naive_bayes_result = "Fake News" if naive_bayes_prediction == 0 else "Real News"


# ==========================================
# 6. DISPLAY FINAL RESULTS
# ==========================================

print("\n========== PREDICTION RESULTS ==========")

print("\nLogistic Regression:")
print("Prediction:", logistic_result)
print("Confidence:", round(logistic_confidence, 2), "%")

print("\nNaive Bayes:")
print("Prediction:", naive_bayes_result)
print("Confidence:", round(naive_bayes_confidence, 2), "%")

print("\n========================================")
print("Note: These are ML model predictions,")
print("not independent verification of news.")
print("========================================")
