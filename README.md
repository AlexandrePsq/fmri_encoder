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

```python
import numpy as np
from fmri_encoder.encoder import Encoder

n_samples_train = 1000
n_samples_test = 100
n_features = 300
n_voxels = 15000
X_train = np.random.random((n_samples_train, n_features))
X_test = np.random.random((n_samples_test, n_features))
Y_train = np.random.random((n_samples_train, n_voxels))
Y_test = np.random.random((n_samples_test, n_voxels))

# Instantiating the encoding model
linearmodel = 'ridgecv'
encoding_params = {
    "alpha": 0.001,
    "alpha_per_target": True,
}
encoder = Encoder(
    linearmodel=linearmodel, saving_folder=None, **encoding_params
)
encoder.fit(X_train, Y_train)

# Testing on test set
test_predictions = encoder.predict(X_test)
score = encoder.eval(test_predictions, Y_test, axis=0)
```

Summarized as:

```python
from fmri_encoder.lazy import default_encoder

output_dict = default_encoder(X_train, Y_train, X_test=X_test, Y_test=Y_test)
# output_dict = {"encoder": encoder, "score": score, "predictions": predictions}
```

### 2) Cross-validation with runs

```python
import numpy as np
from fmri_encoder.encoder import Encoder
from sklearn.model_selection import LeavePOut

# Variables
n_samples = 1000
n_features = 300
n_voxels = 15000
n_runs = 9
out_per_fold = 1
linearmodel = 'ridgecv'
encoding_params = {
    "alpha": 0.001,
    "alpha_per_target": True,
}
# Instantiation
X = [np.random.random((n_samples, n_features)) for i in range(n_runs)]
Y = [np.random.random((n_samples, n_voxels)) for i in range(n_runs)]
logo = LeavePOut(out_per_fold)
encoder = Encoder(
    linearmodel=linearmodel, saving_folder=None, **encoding_params
)
scores = []
# Loop
for train, test in logo.split(X):
    Y_train = np.vstack([Y[i] for i in train])
    X_train = np.vstack([X[i] for i in train])
    Y_test = np.vstack([Y[i] for i in test])
    X_test = np.vstack([X[i] for i in test])
    # Fitting
    encoder.fit(X_train, Y_train)
    # Prediction
    test_predictions = encoder.predict(X_test)
    # Evaluation
    score = encoder.eval(test_predictions, Y_test, axis=0)
    scores.append(score)
cv_score = np.mean(np.stack(scores, axis=0), axis=0)
```

Summarized as:

```python
from fmri_encoder.lazy import default_cv_encoder

output_dict = default_cv_encoder(X, Y, return_preds=False)
# output_dict = {"scores": scores, "cv_score": cv_score, "predictions": predictions}
```

### 3) Aligning features and fMRI brain data

Load all data from `https://osf.io/73nvu/`, and put everything into a folder named `data` at the root of this Github repository.

```python
import os, glob
import numpy as np
import pandas as pd
from fmri_encoder.encoder import Encoder
from sklearn.model_selection import LeavePOut
from fmri_encoder.utils import check_folder
from fmri_encoder.features import FMRIPipe, FeaturesPipe

# Variables
ROOT = '...' # fill here
output_folder = os.path.join(ROOT, 'derivatives')
check_folder(output_folder)

# Parameters for alignment
tr = 2
encoding_method = 'hrf' # 'fir'

# Load the onset/offset of the events
offsets = sorted(glob.glob(os.path.join(ROOT, 'data/word*')))
offsets = [pd.read_csv(p)['offsets'].values for p in offsets]
# Load features
X = sorted(glob.glob(os.path.join(ROOT, 'data/activations*')))
X = [pd.read_csv(p).values for p in X]
# Load fMRI brain data
Y = sorted(glob.glob(os.path.join(ROOT, 'data/fMRI*')))

assert len(offsets)==len(X)
assert len(Y)==len(X)

# Instantiating the fMRI data processing pipeline
fmri_pipe = FMRIPipe(
    fmri_reduction_method=None, 
    fmri_ndim=None
    )
# Instantiating the features processing pipeline
features_reduction_method = None # you can reduce the dimension if you want: 'pca', ...
features_ndim = None # 100, ...
features_pipe = FeaturesPipe(
        features_reduction_method=features_reduction_method, 
        features_ndim=features_ndim
        )

# Fetch or create a masker object that retrieve the voxels of interest in the brain
masker_path = os.path.join(output_folder, 'masker') # path without the extension !! 
masker = fetch_masker(masker_path, Y, **{'detrend': True, 'standardize': True})

# Preprocess fmri data with the masker
Y = [masker.transform(f) for f in Y]
nscans = [f.shape[0] for f in Y] # Number of scans per session
Y = [fmri_pipe.fit_transform(y) for y in Y]

# Preprocess features
X = [
    features_pipe.fit_transform(
        x,
        encoding_method=encoding_method,
        tr=tr,
        groups=get_groups([offset_x]),
        gentles=[offset_x],
        nscans=[nscan_x],
    )
    for (x, offset_x, nscan_x) in zip(X, offsets, nscans)
]
```


Summarized as:

```python
from fmri_encoder.lazy import default_processing

masker_path = "masker" # put the path to your masker (without the extension: '.nii.gz', '.yml', ...)
output_dict = default_processing(X, Y, offsets, tr)
# output_dict = {"X": X, "Y": Y, "masker": masker}
```


### 4) All in one


Summarized as:

```python
from fmri_encoder.lazy import default_process_and_cv_encode

# X list of np.Arrays
# Y list of np.Arrays
# offsets list of np.Arrays
masker_path = "masker" # put the path to your masker (without the extension: '.nii.gz', '.yml', ...)
output_dict = default_process_and_cv_encode(
    X, Y, offsets, tr, return_preds=False, masker_path=masker_path
)
# output_dict = {"scores": scores, "cv_score": cv_score, "predictions": predictions}
```




### 5) Visualizing results

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

To cite this work, please, put a star on the repo and use:

```python
@misc{pasquiou2022neural,
      title={Neural Language Models are not Born Equal to Fit Brain Data, but Training Helps}, 
      author={Alexandre Pasquiou and Yair Lakretz and John Hale and Bertrand Thirion and Christophe Pallier},
      year={2022},
      eprint={2207.03380},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```