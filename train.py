import pandas as pd
import re
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------- LOAD & PREPARE DATA ----------------
df = pd.read_csv(r"D:\PROJECTS\INT234\stress.csv")
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODELS & PARAM GRIDS ----------------
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=500),
        "params": {"C": [0.1, 1, 5]},
    },
    "Linear SVM": {
        "model": LinearSVC(),
        "params": {"C": [0.1, 1, 5]},
    },
    "Naive Bayes": {
        "model": MultinomialNB(),
        "params": {"alpha": [0.1, 0.5, 1.0]},
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {"n_estimators": [150, 300], "max_depth": [None, 20]},
    },
}

best_model = None
best_accuracy = 0
best_name = ""

results = []   # store {name, accuracy}

# ---------------- TRAIN & EVALUATE ----------------
for name, config in models.items():
    print("\n==============================")
    print(f"🔹 Training {name}")
    print("==============================")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid.fit(X_train_vec, y_train)
    model = grid.best_estimator_
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    results.append({"model": name, "accuracy": acc})

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# ---------------- SAVE BEST MODEL & VECTORIZER ----------------
joblib.dump(best_model, "stress_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ TRAINING COMPLETE")
print(f"🏆 Best Model: {best_name}")
print(f"🎯 Best Accuracy: {best_accuracy}")
print("💾 Model & Vectorizer Saved")

# ---------------- VISUALIZATIONS ----------------
results_df = pd.DataFrame(results)

# 1) Line graph: model vs accuracy
plt.figure(figsize=(8, 4))
plt.plot(results_df["model"], results_df["accuracy"], marker="o")
plt.title("Model Accuracy Comparison (Line Graph)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Bar chart of accuracies
plt.figure(figsize=(8, 4))
plt.bar(results_df["model"], results_df["accuracy"], color="skyblue")
plt.title("Model Accuracy Comparison (Bar Chart)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 3) Confusion-matrix heatmap for BEST model
best_preds = best_model.predict(X_test_vec)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_preds,
    cmap="Blues",
    colorbar=True,
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()

# 4) Top “stress” words bar chart for BEST model (if linear)
if hasattr(best_model, "coef_"):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = best_model.coef_[0]

    # top positive (stress) words
    top_pos_idx = np.argsort(coefs)[-10:]
    top_pos_words = feature_names[top_pos_idx]
    top_pos_values = coefs[top_pos_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(top_pos_words, top_pos_values, color="tomato")
    plt.title(f"Top Words Indicating Stress - {best_name}")
    plt.xlabel("Coefficient weight")
    plt.tight_layout()
    plt.show()
