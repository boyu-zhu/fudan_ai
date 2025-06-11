import random
import warnings
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")



class shapley(object):
    def __init__(self, X, f):
        self.X = X
        self.f = f
        self.n_features = X.shape[-1]
        self.features = list(range(X.shape[-1]))
        
    def estimate(self, x, j_=None, M=1000):
        J  = self.features if j_ is None else [ j_ ]
        MC = [ ]
        for j in J:
            MC_j    = [ ]
            X_not_j = [ _ for _ in self.features if _ != j ]
            for m in range(M):
                # sample z
                z = self.X.sample(1).values[0]
                # sample n_K
                #n_K = np.random.choice(X_not_j)
                n_K = np.random.randint(self.n_features)
                # sample K(n_K)
                K = list(np.random.choice(X_not_j, n_K, replace=False))
                #K = list(np.random.choice(X_not_j, np.random.choice(X_not_j), replace=False))
                # create pseudo sample(s)
                xpj = np.array([ x[k] if k in K + [j] else z[k] for k in self.features ]).reshape(1, self.n_features)
                xmj = np.array([ x[k] if k in K       else z[k] for k in self.features ]).reshape(1, self.n_features)
                # calculate marginal contribution of j for iteration m
                MC_j += [ self.f(xpj) - self.f(xmj) ]
            # average to get the monte carlo estimate
            MC   += [ np.mean(MC_j) ]
        
        return np.array(MC)   