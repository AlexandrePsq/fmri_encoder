# Encoding Tutorial for MAIN Conference - Montreal 2023

Repository containing several functions/classes to 
* 1) extract features from a text using a GloVe or a GPT-2 model.
* 2) use these features to fit fMRI brain data.


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

glove_model = load_model_and_tokenizer()
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
from src.data import load_fmri_data, load_stmuli, fetch_masker

# Fetch fmri and stimuli data
fmri_data = load_fmri_data(fmri_url, download=True, template='')
stimuli = load_stmuli(stimuli_url, download=True, template='')

# Fetch or create a masker object that retrieve the voxels of interest in the brain
masker = fetch_masker('masker', fmri_data, **{'detrend': False, 'standardize': False})

# Process fmri data with the masker
fmri_data = preprocess_fmri_data(fmri_data, masker)
```


### Creating the encoding pipeline

```python
from src.encoder import Encoder

fmri_ndim = None
features_ndim = 50
reduction_method = 'pca'
tr = 1.49
encoding_method = 'hrf'
linearmodel = 'ridgecv'

encoder = Encoder(
    linearmodel=linearmodel, 
    reduction_method=reduction_method, 
    fmri_ndim=fmri_ndim, 
    features_ndim=features_ndim, 
    encoding_method=encoding_method, 
    tr=tr
    )

```

### Training encoder

```python

# Extracting features
features_gpt2 = [
    extract_features(
        s['word'].values, 
        gpt2_model, 
        tokenizer,
        FEATURE_COUNT=768,
        NUM_HIDDEN_LAYERS=12,
        ) for s in stimuli
    ] # list of pandas DataFrames

lengths = [len(df) for df in features_gpt2]

start_stop = []
start = 0
for l in lengths:
    stop = start + l
    start_stop.append((start, stop))
    start = stop

nscans = [f.shape[0] for f in fmri_data]
gentles = [s['offsets'].values for s in stimuli]
groups = [np.arange(start, stop, 1) for (start, stop) in start_stop]
Y = np.vstack(fmri_data)

# Computing R maps for GloVe
features_glove = [df.values for sf in features_glove]
X_glove = np.vstack(features_glove) # shape: (#words_total * #features)

encoder.fit(X_glove, Y, groups=groups, gentles=gentles, nscans=nscans)
pred = encoder.predict(X_glove)
scores_glove = encoder.eval(pred, Y)


# Computing R maps for GPT-2
features_gpt2 = [df.values for sf in features_gpt2]
X_gpt2 = np.vstack(features_gpt2) # shape: (#words_total * #features)

encoder.fit(X_gpt2, Y, groups=groups, gentles=gentles, nscans=nscans)
pred = encoder.predict(X_gpt2)
scores_gpt2 = encoder.eval(pred, Y)
```


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
