import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(np.array(X), np.array(y))

    def _gini(self, y):
        classes = np.unique(y)
        return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

    def _best_split(self, X, y):
        m, n = X.shape
        best_gain = -1
        split_idx, split_thr = None, None
        parent_gini = self._gini(y)

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue
                gini = (len(left) / len(y)) * self._gini(left) + (len(right) / len(y)) * self._gini(right)
                gain = parent_gini - gini
                if gain > best_gain:
                    best_gain = gain
                    split_idx, split_thr = feature, threshold
        return split_idx, split_thr

    def _grow_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return DecisionTreeNode(value=np.mean(y))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return DecisionTreeNode(value=np.bincount(y).argmax())

        left_idxs = X[:, feat] <= thr
        right_idxs = X[:, feat] > thr
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return DecisionTreeNode(feature=feat, threshold=thr, left=left, right=right)

    def _predict(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in np.array(X)])
