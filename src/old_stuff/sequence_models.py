"""
Comparing different sequence models
"""
import re
import string
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable




""" Define Models """

""" Simple LSTM Model """
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, h_dim, n_layers=1):
        super(LSTMModel, self).__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        self.char2int = None
        self.int2char = None
        
        self.lstm = nn.LSTM(vocab_size, h_dim, n_layers, dropout=0, batch_first=True)
        self.decoder = nn.Linear(h_dim, vocab_size)
    
    def forward(self, x, state):
        lstm_out, h = self.lstm(x, state)
        lstm_out = lstm_out.contiguous().view(-1, self.h_dim)
        y = F.softmax(self.decoder(lstm_out))
        return y, h
    
    """ Initial hidden state (a and c) """
    def init_state(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.h_dim),
                torch.zeros(self.n_layers, batch_size, self.h_dim))



#%%

# Prepare data
cleaned_text, token_types = get_cleaned_test('grimms_fairy_tales.txt')
train_text, valid_text, test_text = split_data(cleaned_text, num_train_stories=4, num_train_stories=1)

train_encoded = encode_text(train_text, token_types)
valid_encoded = encode_text(valid_text, token_types)
test_encoded = encode_text(test_text, token_types)

#%%

SEQ_LEN = 10
train_x, train_y = prepare_sequences(train_encoded, SEQ_LEN)
valid_x, valid_y = prepare_sequences(valid_encoded, SEQ_LEN)

#%%

# Initialize model
model = LSTMModel(len(token_types), h_dim=20)
initial_hidden = model.init_state()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#%%

""" Train and Evaluation Methods """

def train(model, n_epochs, train_set, valid_set, loss_fn, optimizer, clip=5):
    
    model.train()
    epoch_losses = []
    
    for epoch in range(0, n_epochs):
        print("Epoch {}/{}".format(epoch+1, n_epochs))
        
        hidden = model.init_state()
        for i in range(train_set[0].shape[0]):
            inputs = torch.tensor(train_set[0][i]).float()
            labels = torch.tensor(train_set[1][i]).long()
            
            optimizer.zero_grad()
            
            # Compute output
            output, hidden = model(torch.unsqueeze(inputs, dim=0), hidden)
            
            # Compute loss
            loss = loss_fn(output, labels)
            
            # Compute and update gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            # Create a new variable detached from previous graph for the hidden state
            hidden = tuple(([Variable(var.data) for var in hidden]))
            
            # Print out loss
            if i % 500 == 0:
                print("\t ({}/{}) loss: {}".format(i, train_set[0].shape[0], loss.item()))
                epoch_losses.append(loss.item())
    
    return epoch_losses

def generate(model, initial_text, seq_len):
    model.eval()
    
    sequence = torch.tensor(np.array([text_to_encoded(initial_text, token_types)])).float()
    hidden = model.init_state()
    
    # Loop through and generate the specified number of characters
    generated = []
    for i in range(seq_len + 1):
        # Generate next character from given sequence and hidden
        sequence, hidden = model(sequence, hidden)
        
        # Set new sequence to last generated character
        sequence = torch.unsqueeze(torch.unsqueeze(sequence[-1], 0), 0)
        
        # Add to generated text
        generated.append(sequence.detach().numpy()[-1])
    
    # Return initial text concatenated with generated text
    return initial_text + encoded_to_text(np.array(generated), token_types)

#%%

losses = train(model, 10, (train_x, train_y), (valid_x, valid_y), loss_fn, optimizer, clip=5)

plt.plot(losses)

#%%

generated_story = generate(model, "there once lived ", 50)
print(generated_story)


