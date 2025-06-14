import numpy as np
from models.randomtree import RandomTreeModel

class RandomForestModel:
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = np.random.RandomState(random_state)
        self.trees = []
        self.bootstrap_indices = []

    def fit(self, X, y):
        self.trees = []
        self.bootstrap_indices = []

        for i in range(self.n_estimators):
            # bootstrap 采样
            indices = self.random_state.choice(len(X), len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            self.bootstrap_indices.append(indices)

            tree = RandomTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state.randint(0, 10000)  # 保证树之间差异
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_estimators, n_samples)
        # 多数投票
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes
