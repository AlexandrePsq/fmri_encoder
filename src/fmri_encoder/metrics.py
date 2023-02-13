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
        "r": r,
        "r_nan": r_nan,
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

def r(X, Y):
    """Compute the pearson correlation between X and Y.
    Args:
        - X: np.Array
        - Y: np.Array
    """
    if (X.ndim == 1) and (Y.ndim == 1):
        return pearsonr(X, Y)[0]
    elif ((X.ndim == 1) and ~(Y.ndim == 1)) | (~(X.ndim == 1) and (Y.ndim == 1)):
        raise ValueError(f'Dimension Mismatch between X: {X.shape} and Y: {Y.shape}')
    elif (X.ndim == 2) and (Y.ndim == 1):
        return np.hstack([pearsonr(X[:,i], Y)[0] for i in range(X.shape[1])])
    elif (X.ndim == 2) and (Y.ndim == 2):
        assert X.shape[1]==Y.shape[1]
        return np.hstack([pearsonr(X[:,i], Y[:, i])[0] for i in range(X.shape[1])])
    else:
        raise ValueError(f'Input dimension X: {X.shape} and Y: {Y.shape}. Please apply ‘r‘ function on matrices with ndim=2. Correlation will be computed on axis 0.')

#def r(X, Y):
#    """Compute the pearson correlation between X and Y.
#    Args:
#        - X: np.Array (#bsz, n_samples, #voxels)
#        - Y: np.Array (n_samples, #voxels)
#    """
#    if X.ndim == 1:
#        X = X[None, :, None]
#    elif X.ndim == 2:
#        X = X[None, ...]
#    if Y.ndim == 1:
#        Y = Y[None, :, None]
#    elif Y.ndim == 2:
#        Y = Y[None, ...]
#    X = X - X.mean(1)[:, None, :]
#    Y = Y - Y.mean(1)[:, None, :]
#    SX2 = (X**2).sum(1) ** 0.5
#    SY2 = (Y**2).sum(1) ** 0.5
#    SXY = (X * Y).sum(1)
#    return SXY / (SX2 * SY2)

def r_nan(X, Y):
    """Compute the pearson correlation between X and Y.
    Args:
        - X: np.Array
        - Y: np.Array
    Returns:
        - out: np.Array
    """
    if (X.ndim == 1) and (Y.ndim == 1):
        mask = np.isnan(X) | np.isnan(Y)
        return pearsonr(X[~mask], Y[~mask])[0]
    elif ((X.ndim == 1) and ~(Y.ndim == 1)) | (~(X.ndim == 1) and (Y.ndim == 1)):
        raise ValueError(f'Dimension Mismatch between X: {X.shape} and Y: {Y.shape}')
    elif (X.ndim == 2) and (Y.ndim == 1):
        output = np.hstack([pearsonr(
            X[:,i][~(np.isnan(X[:,i]) | np.isnan(Y))], 
            Y[~(np.isnan(X[:,i]) | np.isnan(Y))]
            )[0] for i in range(X.shape[1])])
        return output
    elif (X.ndim == 2) and (Y.ndim == 2):
        assert X.shape[1]==Y.shape[1]
        output = np.hstack([pearsonr(
            X[:,i][~(np.isnan(X[:,i]) | np.isnan(Y[:,i]))], 
            Y[:,i][~(np.isnan(X[:,i]) | np.isnan(Y[:,i]))]
            )[0] for i in range(X.shape[1])])
        return output
    else:
        raise ValueError(f"Input dimension X: {X.shape} and Y: {Y.shape}. Please apply ‘r‘ function on matrices with ndim=2. Correlation will be computed on axis 0.")

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
