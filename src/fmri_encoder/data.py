import os

import nibabel as nib
from nilearn import image, maskers, masking
from fmri_encoder.utils import read_yaml, save_yaml



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
    params = read_yaml(path + '.yml')
    mask_img = nib.load(path + '.nii.gz')
    if resample_to_img_ is not None:
        mask_img = image.resample_to_img(mask_img, resample_to_img_, interpolation='nearest')
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
    params = masker.get_params()
    params = {key: params[key] for key in ['detrend', 'dtype', 'high_pass', 'low_pass', 'mask_strategy', 
                                            'memory_level', 'smoothing_fwhm', 'standardize',
                                            't_r', 'verbose']}
    nib.save(masker.mask_img_, path + '.nii.gz')
    save_yaml(params, path + '.yml')

def fetch_masker(masker_path, fmri_data, **kwargs):
    """ Fetch or compute if needed a masker from fmri_data.
    Args:
        - masker_path: str
        - fmri_data: list of NifitImages/str
    Returns:    
        - masker: NifitMasker
    """
    if os.path.exists(masker_path + '.nii.gz') and os.path.exists(masker_path + '.yml'):
        masker = load_masker(masker_path, **kwargs)
    else:
        masks = [masking.compute_epi_mask(f) for f in fmri_data]
        mask = image.math_img('img>0.5', img=image.mean_img(masks)) # take the average mask and threshold at 0.5
        masker = maskers.NiftiMasker(mask, **kwargs)
        masker.fit()
        save_masker(masker, masker_path)
    return masker

##################
## Processing data
##################

def intersect_binary(img1, img2):
    """ Compute the intersection of two binary nifti images.
    Args:
        - img1: NifitImage
        - img2: NifitImage
    Returns:
        - intersection: NifitImage
    """
    intersection = image.math_img('img==2', img=image.math_img('img1+img2', img1=img1, img2=img2))
    return intersection
