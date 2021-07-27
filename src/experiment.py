import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import string
import re
import matplotlib.pyplot as plt

from utils import *
from models import *
from run_model import * 

#%%

END_OF_STORY_TOKEN = "#"
VALID_PROP = 0.2

#%%

# read and prepare the data
with open('data/grimms_fairy_tales.txt', 'r') as f:
    text = f.read()

# Init tokenizer and tokenize data
tokenizer = Tokenizer(non_alpha_chars=END_OF_STORY_TOKEN)

# Clean text
text = clean_text(text, tokenizer.chars)

# Split text
train_text, valid_text = split_text_seq(text, END_OF_STORY_TOKEN, 1 - VALID_PROP)

# Tokenize
train_tokenized = np.array(tokenizer.tokenize(train_text))
valid_tokenized = np.array(tokenizer.tokenize(valid_text))

#%%

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU')
else: 
    print('No GPU available')
 
#%%

model = CharLSTM(n_tokens=len(tokenizer.chars), n_hidden=25, n_layers=2)

#%%

t_model = TransformerModel(len(tokenizer.chars), ninp=32, nhead=2, nhid=30, nlayers=1, dropout=0.2)


#%%

batch_size = 2
seq_length = 10 # max length verses

train_x, train_y = get_batches(train_tokenized, batch_size, seq_length)
valid_x, valid_y = get_batches(valid_tokenized, batch_size, seq_length)

#%%

# train the model
print("Training...")
losses = train_lstm(model, train_x, train_y, tokenizer, n_epochs=10, lr=0.001, print_every=1000, train_on_gpu=train_on_gpu)
plt.plot(losses)
print("Done")

#%%

# train the model
print("Training...")
losses = train_transformer(t_model, train_x, train_y, tokenizer, n_epochs=10, lr=0.001, print_every=1000, train_on_gpu=train_on_gpu)
plt.plot(losses)
print("Done")

#%%

print("Evaluating...")
print("Training Perplexity  : ", evaluate(model, train_x, train_y, tokenizer))
print("Validation Perplexity: ", evaluate(model, valid_x, valid_y, tokenizer))
#%%

print("Evaluating...")
print("Training Perplexity  : ", evaluate_transformer(t_model, train_x, train_y, tokenizer))
print("Validation Perplexity: ", evaluate_transformer(t_model, valid_x, valid_y, tokenizer))
#%%

print(sample_lstm(model, 1000, tokenizer, prime='a', top_k=2, train_on_gpu=train_on_gpu))

#%%

print(sum(p.numel() for p in t_model.parameters() if p.requires_grad))

