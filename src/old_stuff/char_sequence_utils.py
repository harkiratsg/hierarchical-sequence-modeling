import re
import string
import matplotlib.pyplot as plt
import numpy as np


""" Get character sequence from file """
def get_cleaned_text(filename, non_alpha_tokens):
    text = ""
    with open(filename, 'r') as reader:
        text += reader.read() + " "
    
    # Lowercase text and remove multiple spaces
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    
    # Only keep alphabet, spaces, and special tokens
    token_types = list(string.ascii_lowercase + " " + non_alpha_tokens)
    
    cleaned_text = ""
    for c in text:
        if c in token_types:
            cleaned_text += c
    return cleaned_text, token_types

""" Create train validation split """
def split_sequences(text, seq_sep_token, num_train_seq, num_valid_seq):
    sequences = text.split(seq_sep_token)
    
    train = seq_sep_token.join(sequences[:num_train_seq]) + seq_sep_token
    valid = seq_sep_token.join(sequences[num_train_seq:num_train_seq + num_valid_seq]) + seq_sep_token
    test = seq_sep_token.join(sequences[num_train_seq + num_valid_seq:]) + seq_sep_token
    
    return train, valid, test

""" Convert text to encoded """
def text_to_encoded(text, token_types):
    return ids_to_encoded(text_to_ids(text, token_types), token_types)

""" Convert encoded to text """
def encoded_to_text(encoded, token_types):
    return ids_to_text(encoded_to_ids(encoded, batched=False), token_types)



""" Convert encoded to text """
def ids_to_text(ids, token_types):
    text = ""
    for i in ids:
        text += token_types[i]
        
    return text

""" Convert text to token id list """
def text_to_ids(text, token_types):
    ids = []
    for c in text:
        ids.append(token_types.index(c))
        
    return ids



""" Convert token ids to one hot encoded array """
def ids_to_encoded(ids, token_types):
    encoded = np.zeros((len(ids), len(token_types)))
    encoded[np.arange(len(ids)), ids] = 1
    return encoded

""" Convert one hot encoded array to token id list """
def encoded_to_ids(encoded, batched):
    return np.argmax(encoded, axis= 2 if batched else 1)



""" Create inputs and labels for the model of specified sequence size """
def prepare_sequences(encoded, seq_len):
    # Shift y forward 1 relative to x and set to same size
    x = encoded[:-1,:]
    y = encoded[1:,:]
    
    # Get number of sequences that can be made from this
    num_seq = int(x.shape[0] / seq_len)
    
    # Reduce x and y size to be split into seq_len evenly
    x = x[:num_seq*seq_len,:]
    y = y[:num_seq*seq_len,:]
    
    # Reshape x and y to be (num_seq, seq_len, vocab_dim)
    vocab_dim = x.shape[1]
    x = x.reshape((num_seq, seq_len, vocab_dim))
    y = y.reshape((num_seq, seq_len, vocab_dim))
    
    return x, y