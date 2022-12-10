import os
import yaml
import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import RidgeCV, LinearRegression

logging.basicConfig(filename='loggings.log', level=logging.INFO)


def check_folder(path):
    """Create adequate folders if necessary.
    Args:
        - path: str
    """
    try:
        if not os.path.isdir(path):
            check_folder(os.path.dirname(path))
            os.mkdir(path)
    except:
        pass

def read_yaml(yaml_path):
    """Open and read safely a yaml file.
    Args:
        - yaml_path: str
    Returns:
        - parameters: dict
    """
    try:
        with open(yaml_path, 'r') as stream:
            parameters = yaml.safe_load(stream)
        return parameters
    except :
        print("Couldn't load yaml file: {}.".format(yaml_path))
    
def save_yaml(data, yaml_path):
    """Open and write safely in a yaml file.
    Args:
        - data: list/dict/str/int/float
        - yaml_path: str
    """
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
def write(path, text, end='\n'):
    """Write in the specified text file.
    Args:
        - path: str
        - text: str
        - end: str
    """
    with open(path, 'a+') as f:
        f.write(text)
        f.write(end)


class Identity(PCA):
    def __init__(self):
        """Implement identity operator.
        """
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
    return ['ridgecv', 'glm']

def get_possible_reduction_methods():
    """Fetch possible reduction methods.
    Returns:
        - list
    """
    return [None, 'pca', 'agglomerative_clustering']


def get_linearmodel(name, alpha=1, alpha_min=-3, alpha_max=8, nb_alphas=10, cv=5):
    """Retrieve the 
    """
    if name=='ridgecv':
        logging.info(f'Loading RidgeCV, with {nb_alphas} alphas varying logarithimicly between {alpha_min} and {alpha_max}...')
        return RidgeCV(
            np.logspace(alpha_min, alpha_max, nb_alphas),
            fit_intercept=True,
            alpha_per_target=True,
            scoring='r2',
            cv=cv
        )
    elif name=='glm':
        logging.info(f'Loading LinearRegression...')
        return LinearRegression(fit_intercept=True)
    elif not isinstance(name, str):
        logging.warning('The model seems to be custom.\nUsing it directly for the encoding analysis.')
        return name 
    else:
        logging.error(f"Unrecognized model {name}. Please select among ['ridgecv', 'glm] or a custom encoding model.")


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
    elif method=="pca":
        return PCA(n_components=ndim)
    elif method=='agglomerative_clustering':
        return FeatureAgglomeration(n_clusters=ndim)

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