import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import torch
import torch.nn.functional as F


def get_metric(metric_name):
    """Fetch the metric associated with the metric_name.
    Args:
        - metric_name: str
    Returns:
        - metric: built-in function
    """
    metric_dic = {
        "r": corr,
        "r_nan": corr,
        "r2": lambda x, y: r2_score(x, y, multioutput="raw_values"),
        "r2_nan": r2_nan,
        "mse": lambda x, y: mse(x, y, axis=0),
        "cosine_dist": cosine_dist,
        "mse_dist": mse,
    }
    if metric_name in metric_dic.keys():
        return metric_dic[metric_name]
    else:
        return metric_name


def corr(X, Y):
    """Compute the pearson correlation between X and Y.
    Args:
        - X: np.Array
        - Y: np.Array
    """
    mX = X - np.mean(X, axis=0)
    mY = Y - np.mean(Y, axis=0)
    norm_mX = np.sqrt(np.sum(mX ** 2, axis=0, keepdims=True))
    norm_mX[norm_mX==0] = 1.
    norm_mY = np.sqrt(np.sum(mY ** 2, axis=0, keepdims=True))
    norm_mY[norm_mY==0] = 1.

    return np.sum(mX / norm_mX * mY/norm_mY, axis=0)


def r2_nan(X, Y):
    """Compute the R2 coefficients between X and Y, not taken nan values into account.
    Args:
        - X: np.Array
        - Y: np.Array
    Returns:
        - out: np.Array
    """
    if (X.ndim == 1) and (Y.ndim == 1):
        mask = np.isnan(X) | np.isnan(Y)
        return r2_score(X[~mask], Y[~mask])
    elif ((X.ndim == 1) and ~(Y.ndim == 1)) | (~(X.ndim == 1) and (Y.ndim == 1)):
        raise ValueError(f'Dimension Mismatch between X: {X.shape} and Y: {Y.shape}')
    elif (X.ndim == 2) and (Y.ndim == 1):
        output = np.hstack([r2_score(
            X[:,i][~(np.isnan(X[:,i]) | np.isnan(Y))], 
            Y[~(np.isnan(X[:,i]) | np.isnan(Y))]
            ) for i in range(X.shape[1])])
        return output
    elif (X.ndim == 2) and (Y.ndim == 2):
        assert X.shape[1]==Y.shape[1]
        output = np.hstack([r2_score(
            X[:,i][~(np.isnan(X[:,i]) | np.isnan(Y[:,i]))], 
            Y[:,i][~(np.isnan(X[:,i]) | np.isnan(Y[:,i]))]
            ) for i in range(X.shape[1])])
        return output
    else:
        raise ValueError(f"Input dimension X: {X.shape} and Y: {Y.shape}. Please apply ‘r‘ function on matrices with ndim=2. Correlation will be computed on axis 0.")

def mse(X, Y, axis=-1):
    """Compute the Mean Square Error between X and Y.
    Args:
        - X: np.Array
        - Y: np.Array
    Returns:
        - out: np.Array
    """
    out = (X - Y) ** 2
    if axis is not None:
        out = np.nanmean(out, axis=axis)
    return out

def cosine_dist(X, Y, axis=-1):
    """Compute the cosine similarity between X and Y.
    Args:
        - X: np.Array
        - Y: np.Array
    Returns:
        - out: np.Array
    """
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    out = 1 - F.cosine_similarity(X, Y, dim=axis)
    return out.numpy()
