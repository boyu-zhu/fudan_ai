import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider

def assign_clusters(data, means):
    """
    Takes in a n x d data matrix, and a k x d matrix of the means.
    Returns a length-n vector with the index of the closest mean to each data point.
    """
    n, d = data.shape
    k = means.shape[0]
    assert d == means.shape[1], "Means are of the wrong shape"
    out = np.zeros(n)
    for i, x in enumerate(data):
        out[i] = np.argmin(np.linalg.norm(means - x, axis=1))
    return out

def update_means(data, clusters):
    """
    Takes in an n x d data matrix, and a length-n vector of the
    cluster indices of each point.
    Computes the mean of each cluster and returns a k x d matrix of the means.
    """
    n, d = data.shape
    assert len(clusters) == n
    k = len(set(clusters))
    cluster_means = []
    for i in range(k):
        cluster_mean = np.mean(data[clusters == i], axis=0)
        cluster_means.append(cluster_mean)
    return np.array(cluster_means)

def cost(data, clusters, means):
    """
    Computes the sum of the squared distance between each point
    and the mean of its associated cluster
    """
    out = 0
    n, d = data.shape
    k = means.shape[0]
    assert means.shape[1] == d
    assert len(clusters) == n
    for i in range(k):
        out += np.linalg.norm(data[clusters == i] - means[i])
    return out

def k_means_cluster(data, k):
    """
    Takes in an n x d data matrix and parameter `k`.
    Yields the cluster means and cluster assignments after
    each step of running k-means, in a 2-tuple.
    """
    n, d = data.shape
    means = data[np.random.choice(n, k, replace=False)]
    assignments = assign_clusters(data, means)
    while True:
        yield means, assignments
        means = update_means(data, assignments)
        new_assignments = assign_clusters(data, means)
        if np.all(assignments == new_assignments):
            yield means, assignments
            print("Final cost = {}".format(cost(data, assignments, means)))
            break
        assignments = new_assignments


def assign_clusters(data, means):
    """
    Takes in a n x d data matrix, and a k x d matrix of the means.
    Returns a length-n vector with the index of the closest mean to each data point.
    """
    n, d = data.shape
    k = means.shape[0]
    assert d == means.shape[1], "Means are of the wrong shape"
    out = np.zeros(n)
    for i, x in enumerate(data):
        # Set out[i] to be the cluster whose mean is closest to point x

        ### start assign_cluster ###
        out[i] = np.argmin(np.linalg.norm(means - x, axis=1))
        ### end assign_cluster ###
    return out

def update_means(data, clusters):
    """
    Takes in an n x d data matrix, and a length-n vector of the
    cluster indices of each point.
    Computes the mean of each cluster and returns a k x d matrix of the means.
    """
    n, d = data.shape
    assert len(clusters) == n
    k = len(set(clusters))
    cluster_means = []
    for i in range(k):
        # Set `cluster_mean` to be the mean of all points in cluster `i`
        # (Assume at least one such point exists)

        ### start update_means ###
        cluster_mean = np.mean(data[clusters == i], axis=0)
        ### end update_means ###
        cluster_means.append(cluster_mean)
    return np.array(cluster_means)

def cost(data, clusters, means):
    """
    Computes the sum of the squared distance between each point
    and the mean of its associated cluster
    """
    out = 0
    n, d = data.shape
    k = means.shape[0]
    assert means.shape[1] == d
    assert len(clusters) == n
    for i in range(k):
        out += np.linalg.norm(data[clusters == i] - means[i])
    return out

def k_means_cluster(data, k):
    """
    Takes in an n x d data matrix and parameter `k`.
    Yields the cluster means and cluster assignments after
    each step of running k-means, in a 2-tuple.
    """
    n, d = data.shape
    means = data[np.random.choice(n, k, replace=False)]
    assignments = assign_clusters(data, means)
    while True:
        yield means, assignments
        means = update_means(data, assignments)
        new_assignments = assign_clusters(data, means)
        if np.all(assignments == new_assignments):
            yield means, assignments
            print("Final cost = {}".format(cost(data, assignments, means)))
            break
        assignments = new_assignments