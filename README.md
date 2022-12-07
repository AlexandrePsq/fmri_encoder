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

model = load_model_and_tokenizer()
features = extract_features(
    words, 
    model, 
    FEATURE_COUNT=300,
    )
```


### GPT-2

```python
from models.extract_gpt2_features import extract_features, load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer('gpt2')
features = extract_features(
    words, 
    model, 
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


```



### Visualizing results

```python


```
