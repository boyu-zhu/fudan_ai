from collections import Counter
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io
import random
random.seed(246810)
np.random.seed(246810)
eps = 1e-5 # a small number
class DecisionTree:
 def __init__(self, max_depth=3, feature_labels=None):
 self.max_depth = max_depth
 self.features = feature_labels
 self.left, self.right = None, None # for non-leaf nodes
 self.split_idx, self.thresh = None, None # for non-leaf nodes
 self.data, self.pred = None, None # for leaf nodes
 @staticmethod
 def entropy(y):
 pass
 ent = 0
 n = len(y)
 counts = Counter(y)
 for _, num in counts.items():
 p = num / n
 ent -= p * np.log2(p)
 return ent
 @staticmethod
 def information_gain(X, y, thresh):
 pass
 n = len(y)
 left_index = np.where(X < thresh)[0]
 right_index = np.where(X >= thresh)[0]


  y_left = y[left_index]
 y_right = y[right_index]
 ent_after = len(y_left) / n * DecisionTree.entropy(y_left) + len(y_right)
/ n * DecisionTree.entropy(y_right)
 return DecisionTree.entropy(y) - ent_after
 @staticmethod
 def gini_impurity(X, y, thresh):
 # OPTIONAL
 pass
 @staticmethod
 def gini_purification(X, y, thresh):
 # OPTIONAL
 pass
 def split(self, X, y, feature_idx, thresh):
 """
 Split the dataset into two subsets, given a feature and a threshold.
 Return X_0, y_0, X_1, y_1
 where (X_0, y_0) are the subset of examples whose feature_idx-th feature
 is less than thresh, and (X_1, y_1) are the other examples.
 """
 left_index = X[:, feature_idx] < thresh
 right_index = ~left_index
 X_0, X_1 = X[left_index], X[right_index]
 if y is None:
 y_0, y_1 = left_index, right_index
 else:
 y_0, y_1 = y[left_index], y[right_index]
 return X_0, y_0, X_1, y_1
 def find_best_thresh(self, X, y):
 if X.shape[0] < 3:
 pass
 best_gain = -np.inf
 best_feature = None
 best_thresh = None
 for feature in range(X.shape[1]):
 unique_values = np.sort(np.unique(X[:, feature]))
 for i in range(1, len(unique_values)):
 thresh = (unique_values[i-1] + unique_values[i]) / 2
 gain = self.information_gain(X[:, feature], y, thresh)
 if gain > best_gain:
 best_gain = gain
 best_feature = feature
 best_thresh = thresh
 return best_feature, best_thresh

 def fit(self, X, y):