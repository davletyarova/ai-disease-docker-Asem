import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# === Импорт алгоритмов ===
from algo.logistic_regression_scratch import LogisticRegressionScratch
from algo.knn_scratch import KNNClassifier
from algo.naive_bayes_scratch import NaiveBayesScratch
from algo.svm_linear_scratch import LinearSVM
from algo.pca_scratch import PCAScratch
from algo.kmeans_scratch import KMeansScratch
from algo.apriori_scratch import AprioriScratch
from algo.fp_growth import FPGrowth
from algo.decision_tree_stub import DecisionTreeClassifierScratch
from algo.random_forest_stub import RandomForestClassifierScratch
from algo.gradient_boosting_stub import GradientBoostingClassifierScratch
from algo.linear_regression_scratch import LinearRegressionScratch
# === Загрузка и подготовка данных ===
df = pd.read_csv('dataset/Training.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

os.makedirs('new_model', exist_ok=True)
def save_model(model, name):
    path = os.path.join('new_model', f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'✅ Saved: {path}')

# === Обучение и сохранение ===
lr = LogisticRegressionScratch()
lr.fit(X_train.values, y_train)
save_model(lr, 'logistic_regression')

knn = KNNClassifier(k=3)
knn.fit(X_train.values, y_train)
save_model(knn, 'knn')

nb = NaiveBayesScratch()
nb.fit(X_train.values, y_train)
save_model(nb, 'naive_bayes')

svm = LinearSVM()
svm.fit(X_train.values, y_train)
save_model(svm, 'svm')

pca = PCAScratch(n_components=2)
X_pca = pca.fit_transform(X.values)
save_model(pca, 'pca')

kmeans = KMeansScratch(k=3)
kmeans.fit(X.values)
save_model(kmeans, 'kmeans')

tree = DecisionTreeClassifierScratch(max_depth=10)
tree.fit(X_train.values, y_train)
save_model(tree, 'decision_tree')

forest = RandomForestClassifierScratch(n_estimators=5, max_depth=10)
forest.fit(X_train.values, y_train)
save_model(forest, 'random_forest')

gb = GradientBoostingClassifierScratch(n_estimators=5, learning_rate=0.1, max_depth=3)
gb.fit(X_train.values, y_train)
save_model(gb, 'gradient_boosting')

linreg = LinearRegressionScratch()
linreg.fit(X_train.values, y_train)
save_model(linreg, 'linear_regression')

# === Ассоциации ===
binary_data = X.copy()
binary_data[binary_data > 0] = 1
transactions = [list(binary_data.columns[binary_data.iloc[i] == 1]) for i in range(len(binary_data))]

ap = AprioriScratch(min_support=0.1)
ap.fit(transactions)
save_model(ap.get_frequent_itemsets(), 'apriori')

fp = FPGrowth(min_support=0.1)
fp.fit(transactions)
save_model(fp.get_frequent_itemsets(), 'fp_growth')

# === Сохраняем LabelEncoder ===
with open(os.path.join('new_model', 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
print('✅ Saved: label_encoder.pkl')