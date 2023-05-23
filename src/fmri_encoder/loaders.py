import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge

from fmri_encoder.custom_ridge import CustomRidge


class Identity(PCA):
    def __init__(self):
        """Implement identity operator."""
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X):
        return X


def get_possible_linear_models():
    """Fetch possible reduction methods.
    Returns:
        - list
    """
    return ["ridgecv", "ridge", "ols", "customridge"]


def get_possible_reduction_methods():
    """Fetch possible reduction methods.
    Returns:
        - list
    """
    return [None, "pca", "agglomerative_clustering"]


def get_linearmodel(
    name,
    alpha=1,
    alpha_min=-3,
    alpha_max=8,
    nb_alphas=10,
    alpha_per_target=True,
    nscans=None,
):
    """Retrieve the"""
    if name == "ridgecv":
        return RidgeCV(
            np.logspace(alpha_min, alpha_max, nb_alphas),
            fit_intercept=True,
            alpha_per_target=alpha_per_target,
        )
    elif name == "ridge":
        return Ridge(
            alpha,
            fit_intercept=True,
        )
    elif name == "ols":
        return LinearRegression(fit_intercept=True)
    elif name == "customridge":
        return CustomRidge(alpha_min, alpha_max, nb_alphas, nscans)

    elif not isinstance(name, str):
        return name


def get_reduction_method(method, ndim=None):
    """
    Args:
        - method: str
        - ndim: int
    Returns:
        - output: built-in reduction operator
    """
    if method is None:
        return Identity()
    elif method == "pca":
        return PCA(n_components=ndim)
    elif method == "agglomerative_clustering_L2":
        return FeatureAgglomeration(
            n_clusters=ndim, affinity="euclidean", linkage="ward", pooling_func=np.mean
        )
    elif method == "agglomerative_clustering_cosine":
        return FeatureAgglomeration(
            n_clusters=ndim, affinity="cosine", linkage="average", pooling_func=np.mean
        )


def get_groups(gentles):
    """Compute the number of rows in each array
    Args:
        - gentles: list of np.Array
    Returns:
        - groups: list of np.Array
    """
    # We compute the number of rows in each array.
    lengths = [len(f) for f in gentles]
    start_stop = []
    start = 0
    for l in lengths:
        stop = start + l
        start_stop.append((start, stop))
        start = stop
    groups = [np.arange(start, stop, 1) for (start, stop) in start_stop]
    return groups
