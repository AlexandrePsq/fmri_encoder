import os
import logging
import gdown, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin

import nibabel as nib
from nilearn import image, input_data, masking
from nilearn.glm.first_level import compute_regressor

from .utils import get_reduction_method
from .utils import check_folder, read_yaml, save_yaml, write

logging.basicConfig(filename='loggings.log', level=logging.INFO)


###############
## Loading data
###############

def load_fmri_data(path, download=False, template=''):
    """Load fMRI data from path.
    Download it if not already done.
    Args:
        - path: str
        - download: bool
        - template:str
    Returns:
        - fmri_data: list of Nifti files
    """
    if download:
        output = '../data/fmri_data.zip'
        gdown.download(path, output, quiet=False)
        os.system(f'unzip {output}')

    fmri_data = sorted(glob.glob(f'{template}*.nii'))

    return fmri_data

def load_stmuli(path, download=False, template=''):
    """Load stimuli data from path.
    Download it if not already done.
    Args:
        - path: str
        - download: bool
        - template:str
    Returns:
        - stimuli_data: list of np.Array
    """
    if download:
        output = '../data/stimuli_data.zip'
        gdown.download(path, output, quiet=False)
        os.system(f'unzip {output}')

    stimuli_data = sorted(glob.glob(f'{template}*.npy'))

    return stimuli_data

def load_masker(path, resample_to_img_=None, intersect_with_img=False, **kwargs):
    """Given a path without the extension, load the associated yaml anf Nifti files to compute
    the associated masker.
    Arguments:
        - path: str
        - resample_to_img_: Nifti image (optional)
        - intersect_with_img: bool (optional)
        - kwargs: dict
    """
    params = read_yaml(path + '.yml')
    mask_img = nib.load(path + '.nii.gz')
    if resample_to_img_ is not None:
        mask_img = image.resample_to_img(mask_img, resample_to_img_, interpolation='nearest')
        if intersect_with_img:
            mask_img = intersect_binary(mask_img, resample_to_img_)
    masker = input_data.NiftiMasker(mask_img)
    masker.set_params(**params)
    if kwargs:
        masker.set_params(**kwargs)
    masker.fit()
    return masker

def save_masker(masker, path):
    """Save the yaml file and image associated with a masker
    """
    params = masker.get_params()
    params = {key: params[key] for key in ['detrend', 'dtype', 'high_pass', 'low_pass', 'mask_strategy', 
                                            'memory_level', 'smoothing_fwhm', 'standardize',
                                            't_r', 'verbose']}
    nib.save(masker.mask_img_, path + '.nii.gz')
    save_yaml(params, path + '.yml')



##################
## Processing data
##################

def intersect_binary(img1, img2):
    """ Compute the intersection of two binary nifti images.
    Arguments:
        - img1: NifitImage
        - img2: NifitImage
    Returns:
        - intersection: NifitImage
    """
    intersection = image.math_img('img==2', img=image.math_img('img1+img2', img1=img1, img2=img2))
    return intersection

def preprocess_fmri_data(fmri_data, masker, add_noise_to_constant=True):
    """Load fMRI data and mask it with a given masker.
    Preprocess it to avoid NaN value when using Pearson
    Correlation coefficients in the following analysis.
    Returns numpy arrays, by extracting cortex activations 
    using a NifitMasker.
    Args:
        - fmri_data: list of NifitImages/str
        - masker:  NiftiMasker object
    Returns:
        - fmri_data: list of np.Array
    """
    fmri_data = [masker.transform(f) for f in fmri_data]
    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    if add_noise_to_constant:
        for index in range(len(fmri_data)):
            zero = np.zeros(fmri_data[index].shape[0])
            new = zero.copy()
            new[0] += np.random.random()/1000
            fmri_data[index] = np.apply_along_axis(lambda x: x if not np.array_equal(x, zero) else new, 0, fmri_data[index])
    return fmri_data

def fetch_masker(masker_path, fmri_data, **kwargs):
    """ Fetch or compute if needed a masker from fmri_data.
    Arguments:
        - masker_path: str
        - fmri_data: list of NifitImages/str
    """
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        masker = load_masker(masker_path, **kwargs)
    else:
        mask = masking.compute_epi_mask(fmri_data)
        masker = input_data.NiftiMasker(mask, **kwargs)
        masker.fit()
        save_masker(masker, masker_path)
    return masker



##################
## Classes
##################

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=np.nan):
        self.fill_value = fill_value

    def fit(self, X, y=None, mask=None):
        """Learn the features to remove. Use pre-selected features, constant features and features having nan.
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - mask: np.Array
        """
        logging.info(f'Identifying unuseful features: pre-selected features, constant features and features having nan...')
        std = X.std(0)
        self.n_init_features = X.shape[1]
        if mask is None:
            self.mask = np.isnan(std) | (std == 0.0) 
        else:
            self.mask = np.isnan(std) | (std == 0.0) | mask==1
        self.valid_idx = np.where(~self.mask)[0]
        self.n_valid_features = len(self.valid_idx)
        return self

    def transform(self, X, y=None):
        """Remove the identified features learnt when calling the ‘fit‘ module.
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        logging.info(f'Removing unuseful features...')
        return X[:, self.valid_idx]

    def fit_transform(self, X, y=None):
        """Apply ‘.fit‘ and then ‘.transform‘
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X, y=None):
        """Reconstruct the original matrix from its reduced version by fillinf removed features with ‘self.fill_value‘.
        Args:
            - X: np.Array
            - y: np.Array (unused)
        """
        logging.info(f'Reconstructing features...')
        assert X.shape[1] == self.n_valid_features
        out = np.ones((len(X), self.n_init_features)) * self.fill_value
        out[:, self.valid_idx] = X
        return out


class DesignMatrixBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, method='hrf', tr=2, groups=None, gentles=None, nscans=None):
        """Instanciate a class to create design matrices from generated embedding representations.
        Args:
            - method: str
            - tr: float
            - groups: list of list of int (indexes in feature space)
        """
        self.method = method
        self.tr = tr
        self.groups = groups
        self.gentles = gentles
        self.nscans = nscans

    def compute_dm_hrf(self, X, nscan, gentle, oversampling=30):
        """Compute the design matrix using the HRF kernel from SPM.
        Args:
            - X: np.Array (generated embeddings, size: (#samples * #features))
            - nscan: int
            - gentle: np.Array (#samples)
            - oversampling: int
        Returns:
            - dm: np.Array
        """
        dm = np.concatenate(
                    [
                        compute_regressor(
                            exp_condition=np.vstack(
                                (
                                    gentle,
                                    np.zeros(X.shape[0]),
                                    X[:, index],
                                )
                            ),
                            hrf_model="spm",
                            frame_times=np.arange(
                                0.0,
                                nscan * self.tr, #- gentle[0] // 2
                                self.tr,
                            ),
                            oversampling=oversampling,
                        )[
                            0
                        ]  # compute_regressor returns (signal, name)
                        for index in range(X.shape[-1])
                    ],
                    axis=1,
                )
        return dm

    def compute_dm_fir(self):
        """Compute the design matrix using the FIR method.
        """
        raise NotImplementedError()
    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        """Transform input data
        """
        if self.method =='hrf':
            output = []
            for i, group in enumerate(self.groups):
                X_ = X[group, :]
                gentle = self.gentles[i]
                if isinstance(gentle, str):
                    gentle  = np.load(gentle)
                nscan = self.nscans[i]
                output.append(self.compute_dm_hrf(X_, nscan, gentle, oversampling=30))
            X = np.concatenate(output)
        elif self.method =='fir':
            X = self.compute_dm_fir()
        return X
    
    def fit_transform(self, X, y=None):
        """Apply successively ‘.fit‘ and ‘.transform‘.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.fit(X, y)
        return self.transform(X)
            

class DimensionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, method=None, ndim=None, **method_args):
        """Instanciate a dimension reduction operator.
        Args:
            - method: str
            - ndim: int
            - method_args: dict
        """
        self.method = get_reduction_method(method, ndim=ndim)
    
    def fit(self, X, y=None):
        """Fit the dimension reduction operator.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.method.fit(X, y)
    
    def transform(self, X):
        """Transform X using the fitted dimension reduction operator.
        Args:
            - X: np.Array
        """
        return self.method.transform(X)

    def fit_transform(self, X, y=None):
        """Apply successively ‘.fit‘ and ‘.transform‘.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.method.fit(X, y)
        return self.method.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform X using the fitted dimension reduction operator.
        Args:
            - X: np.Array
        """
        return self.method.inverse_transform(X)
