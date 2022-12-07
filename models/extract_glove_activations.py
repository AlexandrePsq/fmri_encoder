import os
import gdown
import numpy as np
import pandas as pd
from tqdm import tqdm




def load_model_and_tokenizer(trained_model='../data/glove.6B.300d.txt'):
    """Load a GloVe model given its name.
    Download Glove weights from URL if not already done.
    Args:
        - trained_model: str
    Returns:
        - model: dict
    """
    if ~os.path.exists(trained_model):
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        output = '../data/glove.zip'
        gdown.download(url, output, quiet=False)
        os.system(f'unzip {output}')
    model = init_embeddings(trained_model=trained_model)
    return model


def init_embeddings(trained_model='../data/glove.6B.300d.txt'):
    """ Initialize an instance of GloVe Dictionary.
    Args:
        - trained_model: str
    Returns:
        - model: dict
    """
    model = {}
    with open(trained_model, 'r', encoding="utf-8") as f: 
        for line in f: 
            values = line.split() 
            word = values[0] 
            vector = np.asarray(values[1:], "float32") 
            model[word] = vector 
    return model


def extract_features(
    words, 
    model, 
    FEATURE_COUNT=300,
    ):
    """Extract the features from GloVe.
    Args:
        - words: list of str
        - model: GloVe model
    """
    features = []
    columns = ['embedding-{}'.format(i) for i in range(1, 1 + FEATURE_COUNT)]
    features = []
    for item in tqdm(words):
        if item not in model.keys():
            item = '<unk>'
        features.append(model[item])

    features = pd.DataFrame(np.vstack(features), columns=columns)
    return features
