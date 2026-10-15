import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# === STEP 1: Load your CSV file ===
df = pd.read_csv('campus_complaints.csv', on_bad_lines='skip')

# (Perform any cleaning steps here if needed, e.g., df.dropna())

# Define your features (text) and target (categories)
X = df['complaint']  # Replace with your actual text column name
y = df['category']        # Replace with your actual category column name

# === STEP 6: Split your data into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20,      # 20% for testing, 80% for training
    stratify=y,          # Preserves category proportions in both sets
    random_state=42      # Keeps the split consistent every time you run it
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# === STEP 7: Convert complaint text into numerical features ===
# TfidfVectorizer converts words/phrases into numerical features based on importance.
# ngram_range=(1, 2) includes both single words and two-word phrases (like "water supply").
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === STEP 8: Train and evaluate the model ===
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train_tfidf, y_train)

print("TF-IDF feature extraction completed!")
print("Logistic Regression model trained successfully!")
# Evaluate the model on the test set
y_pred = model.predict(X_test_tfidf)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Save the trained ML model
import joblib

joblib.dump(model, "campus_model.joblib")

print("Model saved successfully as campus_model.joblib")
import pickle

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the fitted vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")