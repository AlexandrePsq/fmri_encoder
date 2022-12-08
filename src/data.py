import os
import logging
import gdown, glob
import numpy as np
import pandas as pd

import nibabel as nib
from nilearn import image, input_data, masking
from src.utils import read_yaml, save_yaml

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
        output = './data/fmri_data.zip'
        gdown.download(path, output, quiet=False)
        os.system(f'unzip {output} -d ./data/')

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
        - stimuli_data: list of csv
    """
    if download:
        output = './data/stimuli_data.zip'
        gdown.download(path, output, quiet=False)
        os.system(f'unzip {output} -d ./data/')

    stimuli_data = sorted(glob.glob(f'{template}*.csv'))

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

def preprocess_stimuli_data(stimuli_data):
    """Load stimuli data. Preprocess it to lower cases.
    Returns pandas dataframes.
    Args:
        - stimuli_data: list of str
    Returns:
        - stimuli_data: list of np.Array
    """
    stimuli_data_tmp = [pd.read_csv(f) for f in stimuli_data]
    stimuli_data = []
    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    for stimuli in stimuli_data_tmp:
        stimuli['Word'] = list(map(lambda x: x.lower(), stimuli['Word']))  
        stimuli_data.append(stimuli)
    return stimuli_data

def fetch_masker(masker_path, fmri_data, **kwargs):
    """ Fetch or compute if needed a masker from fmri_data.
    Arguments:
        - masker_path: str
        - fmri_data: list of NifitImages/str
    """
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        masker = load_masker(masker_path, **kwargs)
    else:
        masks = [masking.compute_epi_mask(f) for f in fmri_data]
        mask = image.math_img('img>0.5', img=image.mean_img(masks)) # take the average mask and threshold at 0.5
        masker = input_data.NiftiMasker(mask, **kwargs)
        masker.fit()
        save_masker(masker, masker_path)
    return masker
