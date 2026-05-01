import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine + shuffle
data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

# ✅ Combine title + text (safe version)
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")
data["content"] = data["title"] + " " + data["text"]

# ✅ Clean text
data["content"] = data["content"].str.lower()
data["content"] = data["content"].str.replace(r"[^\w\s]", "", regex=True)

# Features & labels
X = data["content"]
y = data["label"]

# ✅ Convert text → numbers
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")
print("Accuracy:", model.score(X_test, y_test))