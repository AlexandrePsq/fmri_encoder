import os
import numpy as np
import pandas as pd



def extract_glove_features(words, model):
  """Extract the features from GloVe.
  Args:
    - words: list of str
    - model: trained glove model (dict)
  """
  output = [model(word) for word in words]
  return output
