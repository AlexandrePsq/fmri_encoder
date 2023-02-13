# Tutorial on fMRI Encoding Models

## 0) Installation

First, install the requirements.

```shell
git clone git@github.com:AlexandrePsq/fmri_encoder.git
cd fmri_encoder
pip install -r requirements.txt
pip install -e .
```

## Usage and examples

### 1) Fitting a matrix of features to a matrix of fMRI data 

Example:
We first instantiate all variables and object classes.
We then load, mask and process the fMRI data.

```python
import os
import numpy as np
from nibabel.testing import data_path
from nilearn.image import concat_imgs
from fmri_encoder.utils import (
    get_groups, 
    check_folder
)
from fmri_encoder.data import fetch_masker
from fmri_encoder.encoder import Encoder
from fmri_encoder.features import FMRIPipe, FeaturesPipe

fmri_ndim = None                    
features_ndim = None                
features_reduction_method = None    # We do not reduce the number of features of X
fmri_reduction_method = None        # We do not reduce the number of voxels of Y
tr = 2                              # Set your TR here
encoding_method = 'hrf'             # Using the standard haemodynamic function
linearmodel = 'ridgecv'

# Generating fake training data
output_folder = './derivatives'                 # Where outputs will be saved
check_folder(output_folder)                     # Creating it
nsamples = 2000  # (use to define random data)  # you don't have to specify it if you have real features
nfeatures = 768  # (use to define random data)  # you don't have to specify it if you have real features
sample_frequency = 0.2 # (idem)                 # you don't have to specify it if you have real features
example_fmri_data = os.path.join(data_path, 'example4d.nii.gz')
fmri_data = [example_fmri_data]                 # list of 4D nifti images paths
gentles = [np.linspace(                         # you should load the real onsets/offsets
    0,                                          # here we are using fake data
    sample_frequency*nsamples, 
    num=nsamples
    )]                                              # list of the offset arrays for each fMRI data file
features = [np.random.rand(nsamples, nfeatures)]  # list of np array

# Instantiating the encoding model
encoder = Encoder(linearmodel=linearmodel, saving_folder=output_folder)

# Instantiating the fMRI data processing pipeline
fmri_pipe = FMRIPipe(
    fmri_reduction_method=fmri_reduction_method, 
    fmri_ndim=fmri_ndim
    )
# Instantiating the features processing pipeline
features_pipe = FeaturesPipe(
        features_reduction_method=features_reduction_method, 
        features_ndim=features_ndim
        )

# Fetch or create a masker object that retrieve the voxels of interest in the brain
masker = fetch_masker(os.path.join(output_folder, 'masker'), fmri_data, **{'detrend': True, 'standardize': True})

# Preprocess fmri data with the masker
fmri_data = [masker.transform(f) for f in fmri_data]
nscans = [f.shape[0] for f in fmri_data]                                  # Number of scans per session
fmri_data = np.vstack(fmri_data)
fmri_data_train = fmri_pipe.fit_transform(fmri_data)
nvoxels = fmri_data_train.shape[-1]  # (use to define random data)  # you don't have to specify it if you have real fMRI data

# Preprocess features
features = np.vstack(features)
features_train = features_pipe.fit_transform(
    features,
    encoding_method=encoding_method,
    tr=tr,
    groups=get_groups(gentles),
    gentles=gentles,
    nscans=nscans)

# Training the encoder
encoder.fit(features_train, fmri_data_train)

# Generating fake testing data

nsamples = 1750  # (use to define random data)  # you don't have to specify it if you have real features
nfeatures = 768  # (use to define random data)  # you don't have to specify it if you have real features
sample_frequency = 0.2 # (idem)                 # you don't have to specify it if you have real features
fmri_data_test = [concat_imgs([example_fmri_data for i in range(3)])]
gentles_test = [np.linspace(                    # you should load the real onsets/offsets
    0,                                          # here we are suing fake data
    sample_frequency*nsamples, 
    num=nsamples
    )]                                                  # list of the offset arrays for each fMRI data file
features_test = [np.random.rand(nsamples, nfeatures)] # list of np array

# Preprocess fmri data with the masker
fmri_data_test = [masker.transform(f) for f in fmri_data_test]
nscans_test = [f.shape[0] for f in fmri_data]                                  # Number of scans per session
fmri_data_test = np.vstack(fmri_data_test)
fmri_data_test = fmri_pipe.fit_transform(fmri_data_test)
nvoxels = fmri_data_test.shape[-1]   # (use to define random data)  # you don't have to specify it if you have real fMRI data

# Preprocess features
features_test = np.vstack(features_test)
features_test = features_pipe.fit_transform(
    features_test,
    encoding_method=encoding_method,
    tr=tr,
    groups=get_groups(gentles_test),
    gentles=gentles_test,
    nscans=nscans_test)

# Testing on a test set
predictions = encoder.predict(features_test)
scores = encoder.eval(predictions, fmri_data_test)
```
The fitted encoder is saved.


### 2) Visualizing results

```python
from fmri_encoder.plotting import pretty_plot

vmax = np.max(scores)
imgs = [masker.inverse_transform(scores)]
zmaps = None
masks = None
names = ['My_first_encoding_model']

pretty_plot(
    imgs, 
    zmaps, 
    masks,
    names,
    ref_img=None,
    vmax=[vmax], 
    cmap='cold_hot',
    hemispheres=['left', 'right'], 
    views=['lateral', 'medial'], 
    categorical_values=None, 
    inflated=False, 
    saving_folder='../derivatives/', 
    format_figure='pdf', 
    dpi=300, 
    plot_name='test',
    row_size_factor=8,          # you can play with these arguments to modify the shape of the brain vertically
    column_size_factor=12,      # you can play with these arguments to modify the shape of the brain horizontally
    overlapping=4,
    )

```

### Citation

To cite this work, use:

```python
@software{Pasquiou_encoding_models_2022,
  author = {Pasquiou Alexandre},
  doi = {},
  month = {12},
  title = {{fMRI Linear Encoding Models}},
  url = {https://github.com/AlexandrePsq/fmri_encoder},
  year = {2023}
}
```