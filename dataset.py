import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def generate_synth_dataset(p: float, n: int, d: int, g: float)->pd.DataFrame:
    """Generating a synthetic dataset
    Args:
        p (float): class prior probability
        n (int): number of observations
        d (int): number of features (dimensions)
        g (float): correlation between features
    Returns:
        Datadrame: dataset with d features, n rows
    """
    Y = np.random.binomial(1, p, size=n)
    S = np.array([[g ** abs(i - j) for j in range(d)] for i in range(d)])
    
    mean_0 = np.zeros(d)
    mean_1 = np.array([1/(i+1) for i in range(d)])
    
    X = np.array([
        multivariate_normal.rvs(mean=mean_1 if y == 1 else mean_0, cov=S)
        for y in Y
    ])
    
    feature_names = [f'f{i+1}' for i in range(d)]
    dataset = pd.DataFrame(X, columns=feature_names)
    dataset['Y'] = Y
    
    return dataset
