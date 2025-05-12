import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# === Загрузка и подготовка данных ===
df = pd.read_csv("dataset/Training.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Путь к сохранённым моделям ===
MODEL_DIR = "new_model"

def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель {name}.pkl не найдена в папке {MODEL_DIR}")
    with open(path, "rb") as f:
        return pickle.load(f)

# === Все алгоритмы, которые нужно протестировать ===
algorithms = {
    "Logistic Regression": "logistic_regression",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Linear Regression": "linear_regression"
}

print("📊 Accuracy на тестовой выборке:\n")

for name, file in algorithms.items():
    try:
        model = load_model(file)
        y_pred = model.predict(X_test.values)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name:25}: ✅ Accuracy = {acc:.4f}")
    except Exception as e:
        print(f"{name:25}: ⚠️ Ошибка — {str(e)}")
