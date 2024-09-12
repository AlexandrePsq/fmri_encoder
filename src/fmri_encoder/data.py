import os

import numpy as np
import nibabel as nib
from nilearn import image, maskers, masking
from fmri_encoder.utils import read_yaml, save_yaml
from fmri_encoder.logger import console


##########
## Maskers
##########


def load_masker(path, resample_to_img_=None, intersect_with_img=False, **kwargs):
    """Given a path without the extension, load the associated yaml anf Nifti files to compute
    the associated masker.
    Args:
        - path: str
        - resample_to_img_: Nifti image (optional)
        - intersect_with_img: bool (optional)
        - kwargs: dict
    Returns:
        - masker: NifitMasker
    """
    console.log(f"Loading masker at {path}.nii.gz...")
    params = read_yaml(path + ".yml")
    mask_img = nib.load(path + ".nii.gz")
    if resample_to_img_ is not None:
        mask_img = image.resample_to_img(
            mask_img, resample_to_img_, interpolation="nearest"
        )
        if intersect_with_img:
            mask_img = intersect_binary(mask_img, resample_to_img_)
    masker = maskers.NiftiMasker(mask_img)
    masker.set_params(**params)
    if kwargs:
        masker.set_params(**kwargs)
    masker.fit()
    return masker


def save_masker(masker, path):
    """Save the yaml file and image associated with a masker
    Args:
        - masker: NifitMasker
        - path: str
    """
    console.log(f"Saving masker at {path}.nii.gz...")
    params = masker.get_params()
    params = {
        key: params[key]
        for key in [
            "detrend",
            "dtype",
            "high_pass",
            "low_pass",
            "mask_strategy",
            "memory_level",
            "smoothing_fwhm",
            "standardize",
            "t_r",
            "verbose",
        ]
    }
    nib.save(masker.mask_img_, path + ".nii.gz")
    save_yaml(params, path + ".yml")


def fetch_masker(masker_path, fmri_data, **kwargs):
    """Fetch or compute if needed a masker from fmri_data.
    Args:
        - masker_path: str
        - fmri_data: list of NifitImages/str
    Returns:
        - masker: NifitMasker
    """
    if os.path.exists(masker_path + ".nii.gz") and os.path.exists(masker_path + ".yml"):
        masker = load_masker(masker_path, **kwargs)
    else:
        console.log(f"Failed to load masker at {masker_path}.nii.gz.")
        console.log(f"Computing masker from fmri_data...")
        masks = [masking.compute_epi_mask(f) for f in fmri_data]
        mask = image.math_img(
            "img>0.5", img=image.mean_img(masks)
        )  # take the average mask and threshold at 0.5
        masker = maskers.NiftiMasker(mask, **kwargs)
        masker.fit()
        save_masker(masker, masker_path)
    return masker


##################
## Processing data
##################


def intersect_binary(img1, img2):
    """Compute the intersection of two binary nifti images.
    Args:
        - img1: NifitImage
        - img2: NifitImage
    Returns:
        - intersection: NifitImage
    """
    intersection = image.math_img(
        "img==2", img=image.math_img("img1+img2", img1=img1, img2=img2)
    )
    return intersection


def process_fmri_data(fmri_paths, masker, language="english", save=False):
    """Load fMRI data from given paths and mask it with a given masker.
    Preprocess it to avoid NaN value when using Pearson
    Correlation coefficients in the following analysis.
    Arguments:
        - fmri_paths: list (of string)
        - masker: NiftiMasker object
    Returns:
        - data: list of length #runs (np.array of shape: #scans * #voxels)
    """
    data = [masker.transform(f) for f in fmri_paths]
    # voxels with activation at zero at each time step generate a nan-value pearson correlation => we add a small variation to the first element
    for run in range(len(data)):
        zero = np.zeros(data[run].shape[0])
        new = zero.copy()
        new[0] += np.random.random() / 1000
        data[run] = np.apply_along_axis(
            lambda x: x if not np.array_equal(x, zero) else new, 0, data[run]
        )
    if save:
        for index, run in enumerate(data):
            np.save(fmri_paths[index].replace(".nii.gz", ".npy"), run.T)
    return data
