import numpy as np

class RandomTreeModel:
    def __init__(self, max_depth=3, min_samples_split=2, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = np.random.RandomState(random_state)
        self.tree = None

    def fit(self, X, y):
        def build_tree(X, y, depth):
            if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
                # 叶节点：返回多数类
                return {'type': 'leaf', 'class': np.bincount(y).argmax()}

            feature = self.random_state.randint(0, X.shape[1])
            threshold = self.random_state.uniform(X[:, feature].min(), X[:, feature].max())
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                return {'type': 'leaf', 'class': np.bincount(y).argmax()}

            left = build_tree(X[left_mask], y[left_mask], depth + 1)
            right = build_tree(X[right_mask], y[right_mask], depth + 1)
            return {
                'type': 'node',
                'feature': feature,
                'threshold': threshold,
                'left': left,
                'right': right
            }

        self.tree = build_tree(X, y, depth=0)
        return self.tree

    def predict_single(self, node, x):
        if node['type'] == 'leaf':
            return node['class']
        if x[node['feature']] < node['threshold']:
            return self.predict_single(node['left'], x)
        else:
            return self.predict_single(node['right'], x)

    def predict(self, X):
        return np.array([self.predict_single(self.tree, x) for x in X])
