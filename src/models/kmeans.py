
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class KMeansModel:
    def __init__(self, n_clusters=3, max_iter=10, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers = None
        self.labels = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        self.centers = X[rng.choice(len(X), self.n_clusters, replace=False)]
        history = []

        for i in range(self.max_iter):
            # 计算每个点到中心的距离，分配标签
            self.labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2), axis=1)

            history.append((self.centers.copy(), self.labels.copy()))

            # 计算新中心
            new_centers = np.array([
                X[self.labels == j].mean(axis=0) if np.any(self.labels == j) else self.centers[j]
                for j in range(self.n_clusters)
            ])

            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers

        return history