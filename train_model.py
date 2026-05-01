import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("📥 Loading data...")

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

print("🔀 Combining data...")

# Combine and shuffle
data = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

print("🧹 Cleaning text...")

# Handle missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Combine title + text
data["content"] = data["title"] + " " + data["text"]

# Clean text
data["content"] = data["content"].str.lower()
data["content"] = data["content"].str.replace(r"[^\w\s]", "", regex=True)

print("🔢 Converting text to numbers (TF-IDF)...")

# Features & labels
X = data["content"]
y = data["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000   # keeps file size small for deployment
)

X = vectorizer.fit_transform(X)

print("✂️ Splitting data...")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🤖 Training model...")

# Model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("💾 Saving model...")

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Accuracy
accuracy = model.score(X_test, y_test)

print("✅ Model trained successfully!")
print(f"🎯 Accuracy: {accuracy:.4f}")