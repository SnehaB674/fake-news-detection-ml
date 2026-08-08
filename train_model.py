import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Keep only required columns
data = data[["text", "label"]]

# Split data
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000)
}

accuracy_results = {}

best_accuracy = 0
best_model = None

print("\n==============================\n")

for name, model in models.items():

    model.fit(X_train_vec, y_train)

    prediction = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, prediction)

    accuracy_results[name] = accuracy

    print(f"{name} Accuracy : {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\n==============================")

print(f"\nBest Model : {type(best_model).__name__}")

# Save best model
joblib.dump(best_model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")

# Plot Accuracy Graph
plt.figure(figsize=(8,5))

plt.bar(
    accuracy_results.keys(),
    [v*100 for v in accuracy_results.values()]
)

plt.ylabel("Accuracy (%)")
plt.title("Machine Learning Algorithm Comparison")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig("models/accuracy_graph.png")

print("Accuracy Graph Saved!")