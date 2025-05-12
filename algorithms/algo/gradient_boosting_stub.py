from collections import Counter
import numpy as np 
from .decision_tree_stub import DecisionTreeClassifierScratch

class GradientBoostingClassifierScratch:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        y_pred = np.full(y.shape, np.bincount(y).argmax())  # инициализация
        for _ in range(self.n_estimators):
            residual = y - y_pred
            tree = DecisionTreeClassifierScratch(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.zeros(len(X))
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return (pred >= 0.5).astype(int)
