import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'new_model')

MODEL_FILES = {
    'logistic_regression.pkl': 'Logistic Regression',
    'decision_tree.pkl': 'Decision Tree',
    'random_forest.pkl': 'Random Forest',
    'gradient_boosting.pkl': 'Gradient Boosting',
    'linear_regression.pkl': 'Linear Regression',
    'naive_bayes.pkl': 'Naive Bayes',
    'knn.pkl': 'KNN',
    'svm.pkl': 'SVM',
    'pca.pkl': 'PCA',
    'kmeans.pkl': 'KMeans',
    'apriori.pkl': 'Apriori',
    'fp_growth.pkl': 'FP-Growth',
}

loaded_models = {}

for filename, display_name in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            loaded_models[display_name] = pickle.load(f)
            print(f"✅ Loaded: {display_name}")
    else:
        print(f"❌ Not found: {filename}")

label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
if os.path.exists(label_encoder_path):
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
        print("✅ Loaded: label_encoder")
else:
    label_encoder = None
    print("❌ label_encoder.pkl not found")
