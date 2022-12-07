import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(name):
    """Load a HuggingFace model and the associated tokenizer given its name.
    Args:
        - name: str
    Returns:    
        - model: HuggingFace model
        - tokenizer: HuggingFace tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer

def pad_to_max_length(sequence, max_seq_length, space=220, special_token_end=50256):
    """Pad sequence to reach max_seq_length
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 220)
        - special_token_end: int (default 50256)
    Returns:
        - result: list of int
    """
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    result = sequence + [space, special_token_end] * ((max_seq_length - n)// 2)
    if len(result)==max_seq_length:
        return result
    else:
        return result + [space]

def create_examples(sequence, max_seq_length, space=220, special_token_beg=50256, special_token_end=50256):
    """Returns list of InputExample objects.
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 220)
        - special_token_beg: int (default 50256)
        - special_token_end: int (default 50256)
    Returns:
        - result: list of int
    """
    return pad_to_max_length([special_token_beg] + sequence + [space, special_token_end], max_seq_length)

def batchify_to_truncated_input(
    iterator, 
    tokenizer, 
    context_size=None, 
    max_seq_length=512, 
    space='Ġ', 
    special_token_beg='<|endoftext|>', 
    special_token_end='<|endoftext|>'
    ):
  """Batchify sentence 'iterator' string, to get batches of sentences with a specific number of tokens per input.
  Function used with 'get_truncated_activations'.
  Arguments:
    - iterator: sentence str
    - tokenizer: Tokenizer object
    - context_size: int
    - max_seq_length: int
    - space: str (default 'Ġ')
    - special_token_beg: str (default '<|endoftext|>')
    - special_token_end: str (default '<|endoftext|>')
  Returns:
    - input_ids: input batched
    - indexes: tuple of int
  """
  max_seq_length = max_seq_length if context_size is None else context_size+5 # +5 because of the special tokens + the current and following tokens
  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  try:
    data = tokenizer.encode(iterator).ids
    text =  tokenizer.encode(iterator).tokens
  except:
    data = tokenizer.encode(iterator)
    text =  tokenizer.tokenize(iterator)

  if context_size==0:
    examples = [create_examples(data[i:i + 2], max_seq_length) for i, _ in enumerate(data)]
    tokens = [create_examples(text[i:i + 2], max_seq_length, space=space, special_token_beg=special_token_beg, special_token_end=special_token_end) for i, _ in enumerate(text)]
  else:
    examples = [create_examples(data[i:i + context_size + 2], max_seq_length) for i, _ in enumerate(data[:-context_size])]
    tokens = [create_examples(text[i:i + context_size + 2], max_seq_length, space=space, special_token_beg=special_token_beg, special_token_end=special_token_end) for i, _ in enumerate(text[:-context_size])]
  # the last example in examples has one element less from the input data, but it is compensated by the padding. we consider that the element following the last input token is the special token.
  features = [torch.FloatTensor(example).unsqueeze(0).to(torch.int64) for example in examples]
  input_ids = torch.cat(features, dim=0)
  indexes = [(1, context_size+2)] + [(context_size+1, context_size+2) for i in range(1, len(input_ids))] # shifted by one because of the initial special token
  # Cleaning
  del examples
  del features
  return input_ids, indexes, tokens

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent, connection_character='Ġ', eos_token='<|endoftext|>'):
  '''Aligns tokenized and untokenized sentence given non-subwords "Ġ" prefixed
  Assuming that each subword token that does start a new word is prefixed
  by "Ġ", computes an alignment between the un-subword-tokenized
  and subword-tokenized sentences.
  Args:
    tokenized_sent: a list of strings describing a subword-tokenized sentence
    untokenized_sent: a list of strings describing a sentence, no subword tok.
  Returns:
    A dictionary of type {int: list(int)} mapping each untokenized sentence
    index to a list of subword-tokenized sentence indices
  '''
  mapping = defaultdict(list)
  untokenized_sent_index = 0
  tokenized_sent_index = 0
  while (untokenized_sent_index < len(untokenized_sent) and tokenized_sent_index < len(tokenized_sent)):
    while (tokenized_sent_index + 1  < len(tokenized_sent) and (not tokenized_sent[tokenized_sent_index + 1].startswith(connection_character)) and tokenized_sent[tokenized_sent_index+1]!=eos_token):
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      tokenized_sent_index += 1
    mapping[untokenized_sent_index].append(tokenized_sent_index)
    untokenized_sent_index += 1
    tokenized_sent_index += 1
  return mapping

def extract_gpt2_features(
    words, 
    model, 
    tokenizer, 
    context_size=100, 
    max_seq_length=512, 
    space='Ġ', 
    bsz=32,
    special_token_beg='<|endoftext|>', 
    special_token_end='<|endoftext|>',
    FEATURE_COUNT=768,
    NUM_HIDDEN_LAYERS=12,
    ):
  """Extract the features from GPT-2.
  Args:
    - words: list of str
    - model: HuggingFace model
    - tokenizer: HuggingFace tokenizer
  """
  hidden_states_activations = []
  iterator = ' '.join(words)
  
  tokenized_text = tokenizer.tokenize(iterator)
  mapping = match_tokenized_to_untokenized(tokenized_text, iterator)
  
  input_ids, indexes, tokens = batchify_to_truncated_input(
      iterator, tokenizer, 
      context_size=context_size, 
      max_seq_length=max_seq_length, 
      space=space, special_token_beg=special_token_beg, 
      special_token_end=special_token_end)

  with torch.no_grad():
    hidden_states_activations_ = []
    for input_tmp in tqdm(input_ids.chunk(input_ids.size(0)//bsz)):
      hidden_states_activations_tmp = []
      encoded_layers = model(input_tmp, output_hidden_states=True)
      hidden_states_activations_tmp = np.stack([i.detach().numpy() for i in encoded_layers.hidden_states], axis=0) #shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
      hidden_states_activations_.append(hidden_states_activations_tmp)
        
    hidden_states_activations_ = np.swapaxes(np.vstack([np.swapaxes(item, 0, 1) for item in hidden_states_activations_]), 0, 1) #shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)
      
  activations = []
  for i in range(hidden_states_activations_.shape[1]):
    index = indexes[i]
    activations.append([hidden_states_activations_[:, i, j, :] for j in range(index[0], index[1])])
  activations = np.stack([i for l in activations for i in l], axis=0)
  activations = np.swapaxes(activations, 0, 1) #shape: (#nb_layers, batch_size, hidden_state_dimension)

  for word_index in range(len(mapping.keys())):
    word_activation = []
    word_activation.append([activations[:, index, :] for index in mapping[word_index]])
    word_activation = np.vstack(word_activation)
    hidden_states_activations.append(np.mean(word_activation, axis=0).reshape(-1))# list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
  #After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)
      
  hidden_states_activations = pd.DataFrame(np.vstack(hidden_states_activations), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + NUM_HIDDEN_LAYERS) for index in range(1, 1 + FEATURE_COUNT)])
  
  return hidden_states_activations


