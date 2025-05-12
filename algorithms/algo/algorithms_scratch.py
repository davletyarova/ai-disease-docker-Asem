import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

# 1. Logistic Regression
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            pred = self._sigmoid(linear)
            dw = (1 / len(X)) * np.dot(X.T, (pred - y))
            db = (1 / len(X)) * np.sum(pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return (self._sigmoid(linear) >= 0.5).astype(int)


# 2. Linear Regression
class LinearRegressionScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(X)) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# 3. K-Nearest Neighbors
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_idxs = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idxs]
        return Counter(k_labels).most_common(1)[0][0]


# 4. Naive Bayes
class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        return np.array([self._classify(x) for x in X])

    def _classify(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
            likelihood -= 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]


# 5. Decision Tree (basic, one-level stub)
class DecisionTreeStub:
    def fit(self, X, y):
        self.majority_class = Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.majority_class)


# 6. Random Forest (ensemble of stubs)
class RandomForestStub:
    def __init__(self, n_trees=5):
        self.n_trees = n_trees
        self.trees = [DecisionTreeStub() for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]


# 7. SVM (Linear Kernel)
class LinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        return np.where(np.dot(X, self.weights) - self.bias >= 0
        )