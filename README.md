# Encoding Tutorial for MAIN Conference - Montreal 2023

Repository containing several functions/classes to 
* 1) extract features from a text using a GloVe or a GPT-2 model.
* 2) use these features to fit fMRI brain data.


## 0) Installation

First install the requirements.

```shell
pip install -r requirements.txt
pip install -e .
```


## 1) Theorical steps

A Neural Language model is used to generate embeddings for each token of a given kind of stimuli (text, image, video). Here we porcess textual information.

Then a linear encoding model (the encoder) is used to fit the model derived representations to fMRI brain data.


## 1) Extracting features with GloVe and GPT-2 

In the folder ‘/models‘, there are two scripts to extract the features from GloVe and GPT-2.
Usecase:

```python
stimuli = pd.read_csv('data/word_run1.csv')
words = stimuli['word].values
```

### GloVe

```python
from models.extract_glove_features import extract_features, load_model_and_tokenizer

glove_model, _ = load_model_and_tokenizer()
features_glove = extract_features(
    words, 
    glove_model, 
    FEATURE_COUNT=300,
    )
```


### GPT-2

```python
from models.extract_gpt2_features import extract_features, load_model_and_tokenizer

gpt2_model, tokenizer = load_model_and_tokenizer('gpt2')
features_gpt2 = extract_features(
    words, 
    gpt2_model, 
    tokenizer,
    FEATURE_COUNT=768,
    NUM_HIDDEN_LAYERS=12,
    )
```

## 2) Creating an Encoding Pipeline to predict fMRI brain data using modelderived features

### Loading and processing fMRI data

```python
from src.data import load_fmri_data, load_stmuli, fetch_masker, preprocess_fmri_data

fmri_url = "https://drive.google.com/file/d/1QsxmYaI-eOG7ip0Lfe82jXJ9-Ip3Oqxy/view?usp=share_link"
stimuli_url = "https://drive.google.com/file/d/11HT-0TH0hOerOkP3zTDzkICqRt7s9ZQZ/view?usp=share_link"

# Fetch fmri and stimuli data
fmri_data = load_fmri_data(fmri_url, download=True, template='')
stimuli = load_stmuli(stimuli_url, download=True, template='')

# Fetch or create a masker object that retrieve the voxels of interest in the brain
masker = fetch_masker('masker', fmri_data, **{'detrend': False, 'standardize': False})

# Process fmri data with the masker
fmri_data = preprocess_fmri_data(fmri_data, masker)
```


## Encoder.py

General python Class to fit linear encoding models.

Example:
We first instantiate all variables and object classes.
We then load, mask and process the fMRI data.

```python
from src.utils import get_groups, preprocess_fmri_data
from src.utils import check_folder, fetch_masker
from src.encoder import Encoder
from src.features import FMRIPipe, FeaturesPipe

fmri_ndim = None
features_ndim = None
features_reduction_method = None #'pca'
fmri_reduction_method = None
tr = ...
encoding_method = 'hrf'
linearmodel = 'ridgecv'

check_folder('./derivatives')
fmri_data = ... # list of 4D nifti images paths
gentles = ... # list of the offsets of the stimuli data
nscans = ... # number of scnas per session
features = ... # list of np array

# Instantiating the encoding model
encoder = Encoder(linearmodel=linearmodel)
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
masker = fetch_masker('./derivatives/masker', fmri_data, **{'detrend': True, 'standardize': True})

# Process fmri data with the masker
fmri_data = preprocess_fmri_data(fmri_data, masker)
fmri_data = np.vstack(fmri_data)
features = np.vstack(features)
```

We then fit the encoding model (with cross-validation for the L2 regularization) but no outside cross-validation because we want to retrieve the trained weights for later decoding.

```python
## Fitting the model with GloVe
# Processing train fMRI data
fmri_data = fmri_pipe.fit_transform(fmri_data)
# Processing Features
features = features_pipe.fit_transform(
    features, 
    encoding_method=encoding_method, 
    tr=tr, 
    groups=get_groups(gentles), 
    gentles=gentles, 
    nscans=nscans)
# Training the encoder
encoder.fit(features, fmri_data)
```
The fitted encoder is saved.



### Visualizing results

```python
from src.plotting import pretty_plot

imgs = [masker.inverse_transform(scores_glove), masker.inverse_transform(scores_gpt2)]
zmaps = None
masks = None
names = ['GloVe', 'GPT-2']

pretty_plot(
    imgs, 
    zmaps, 
    masks,
    names,
    ref_img=None,
    vmax=0.2, 
    cmap='cold_hot',
    hemispheres=['left', 'right'], 
    views=['lateral', 'medial'], 
    categorical_values=False, 
    inflated=False, 
    saving_folder='../derivatives/', 
    format_figure='pdf', 
    dpi=300, 
    plot_name='test',
    row_size_factor=6,
    overlapping=6,
    column_size_factor=12,
    )

```

### Citation

To cite this work, use:

```python
@software{Pasquiou_encoding_2022,
  author = {Pasquiou Alexandre},
  doi = {},
  month = {12},
  title = {{Encoding Pipeline}},
  url = {https://github.com/AlexandrePsq/main_tutorial},
  year = {2022}
}
```