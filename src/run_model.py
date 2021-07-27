import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import string
import re

from utils import *

def train_lstm(model, train_x, train_y, tokenizer, n_epochs=10, lr=0.001, clip=5, val_frac=0.1, print_every=10, train_on_gpu=False):
    # TODO Do batches in here
    
    model.train()
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    if(train_on_gpu):
        model.cuda()
    
    n_chars = len(tokenizer.chars)
    losses = []
    for e in range(n_epochs):
        # initialize hidden state
        hidden = model.init_hidden(train_x.shape[1])
        
        batch_update_steps = 0
        total_epoch_loss = 0
        for x, y in zip(train_x, train_y):
            batch_update_steps += 1
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # zero accumulated gradients
            model.zero_grad()
            
            # get the output from the model
            output, hidden = model(inputs, hidden)
            
            # Backpropagate loss
            loss = loss_fn(output, targets.view(train_x.shape[1] * train_x.shape[2]))
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

            # Clone hidden so it breaks from existing computational graph for next update
            hidden = tuple(([Variable(var.data) for var in hidden]))

            total_epoch_loss += loss.item()
            
        avg_epoch_loss = total_epoch_loss / batch_update_steps
        losses.append(avg_epoch_loss)
        print("Epoch {}/{}: Avg loss: {}".format(e+1, n_epochs, avg_epoch_loss))
    return losses

def train_transformer(model, train_x, train_y, tokenizer, n_epochs=10, lr=0.001, clip=5, val_frac=0.1, print_every=10, train_on_gpu=False):
    # TODO Do batches in here
    
    model.train()
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    if(train_on_gpu):
        model.cuda()
    
    src_mask = model.generate_square_subsequent_mask(train_x.shape[2])
    
    n_chars = len(tokenizer.chars)
    losses = []
    for e in range(n_epochs):
        
        batch_update_steps = 0
        total_epoch_loss = 0
        for x, y in zip(train_x, train_y):
            batch_update_steps += 1
            
            inputs, targets = torch.from_numpy(x).T, torch.from_numpy(y).T
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # zero accumulated gradients
            model.zero_grad()
            
            # get the output from the model
            output = model(inputs, src_mask)
            
            # Backpropagate loss
            loss = loss_fn(output.view(-1, len(tokenizer.chars)), targets.reshape(targets.shape[0] * targets.shape[1]))
            loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()

            total_epoch_loss += loss.item()
            
        avg_epoch_loss = total_epoch_loss / batch_update_steps
        losses.append(avg_epoch_loss)
        print("Epoch {}/{}: Avg loss: {}".format(e+1, n_epochs, avg_epoch_loss))
    return losses

def evaluate(model, eval_x, eval_y, tokenizer):
    model.eval()
    perplexities = []
    
    hidden = model.init_hidden(eval_x.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    
    for x, y in zip(eval_x, eval_y):
        x = one_hot_encode(x, len(tokenizer.chars))
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
        
        predicted_y, hidden = model(inputs, hidden)
        targets = targets.view(y.shape[0] * x.shape[1])
        
        perplexities.append(torch.exp(loss_fn(predicted_y, targets)).item())
        
        # Clone hidden so it breaks from existing computational graph for next update
        hidden = tuple(([Variable(var.data) for var in hidden]))
    
    avg_perplexity = sum(perplexities) / len(perplexities)
    
    return avg_perplexity

def evaluate_transformer(model, eval_x, eval_y, tokenizer):
    model.eval()
    perplexities = []
    
    loss_fn = nn.CrossEntropyLoss()
    src_mask = model.generate_square_subsequent_mask(eval_x.shape[2])
    
    for x, y in zip(eval_x, eval_y):
        inputs, targets = torch.from_numpy(x).T, torch.from_numpy(y).T
        
        # get the output from the model
        output = model(inputs, src_mask)
        
        # Backpropagate loss
        perplexities.append(torch.exp(loss_fn(output.view(-1, len(tokenizer.chars)), targets.reshape(targets.shape[0] * targets.shape[1]))).item())

    avg_perplexity = sum(perplexities) / len(perplexities)
    
    return avg_perplexity

def predict(net, char, tokenizer, h=None, top_k=None, train_on_gpu=False):
        ''' 
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[tokenizer.char2int[char]]])
        
        x = one_hot_encode(x, len(tokenizer.int2char))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving x
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        # considering the k most probable characters with topk method
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return tokenizer.int2char[char], h

def sample_lstm(net, size, tokenizer, prime='Il', top_k=None, train_on_gpu=False):
        
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, tokenizer, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], tokenizer, h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

