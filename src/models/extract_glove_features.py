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
        - tokenizer: None
    """
    if ~os.path.exists(trained_model):
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        output = './data/glove.zip'
        gdown.download(url, output, quiet=False)
        os.system(f'unzip {output}')
    model = init_embeddings(trained_model=trained_model)
    return model, None


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

def update_model(glove_model, embedding_size=300):
    """ Ad some words to a glove model.
    Args:
        - glove_model: dict
    Returns:
        - glove_model: dict
    """
    words2add = {"that's":(['that', "is"], 1),
                        "it's":(['it', "is"], 1),
                        "what's":(['what', "is"], 1),
                        "i'm":(['i', "am"], 1),
                        "can't":(['can', "not"], 1),
                        "you're":(['you', "are"], 1),
                        "we're":(['we', "are"], 1),
                        "don't":(['do', "not"], 1),
                        "buffay":(['she'], 1),
                        "she's":(['she', "has"], 1),
                        "victoria's":(['victoria', "has"], 1),
                        "chandler's":(['chandler', "has"], 1),
                        "dexter's":(['dexter', "has"], 1),
                        "women's":(['women', "has"], 1),
                        "i've":(['i', "have"], 1),
                        "i'll":(['i', "will"], 1),
                        "didn't":(['did', "not"], 1),
                        "wasn't":(['was', "not"], 1),
                        "we've":(['we', "have"], 1),
                        "he's":(['he', "has"], 1),
                        "we've":(['we', "have"], 1),
                        "we've":(['we', "have"], 1),
                        "we've":(['we', "have"], 1),
                        "we've":(['we', "have"], 1),
                        "tonight's": (['tonight', "has"], 1),
                        "blackout's": (['blackout', "has"], 1),
                        "you'd": (['you', "would"], 1),
                        "dad's": (['dad', "has"], 1),
                        "oohhoo": (['oh'], 0),
                        "pheebs": (['she'], 0),
                        "let's": (['let', "us"], 1),
                        "wouldn't": (['would', "not"], 1),
                        "exwife": (['wife'], 0),
                        "there's": (['there', "is"], 1),
                        "mmhmm": (['oh'], 0),
                        "it'snever": (['it', "has", "never"], 2),
                        "you'll": (['you', "will"], 1),
                        "shhshhshh": (['oh'], 0),
                        "lalalalala": (['oh'], 0),
                        "couldn't": (['could', 'not'], 1),
                        "carol's":(['she', "has"], 1),
                        "bloomingdale's":(['city', "has"], 1),
                        "who's":(['who', "has"], 1),
                        "isn't":(['is', "not"], 1),
                        "doesn't":(['does', "not"], 1),
                        "men's":(['men', "has"], 1),
                        "uhhuh":(['oh'], 0),
                        "won't":(['will', "not"], 1),
                        "youwent":(['you', "went"], 1),
                        "brother's":(['brother', "has"], 1),
                        } # the second value in the tuple is the number of following words to skip in generate
    for key in words2add.keys():
        if key not in glove_model.keys():
            glove_model[key] = np.zeros((embedding_size,))
            for word in words2add[key][0]:
                try:
                    glove_model[key] += glove_model[word]
                except:
                    print(f'{word} does not appear in the vocabulary... Be sure that it is normal.')
            glove_model[key] = glove_model[key] / len(words2add[key][0])
    return glove_model

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
