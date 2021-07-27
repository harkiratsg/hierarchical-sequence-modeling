
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from char_sequence_utils import *

#%%

print("reading and encoding...")

text, token_types = get_cleaned_text(filename="data/nested_abcxyz.txt", non_alpha_tokens=",.*#")

train_text, valid_text, test_text = split_sequences(text=text, seq_sep_token="#", num_train_seq=3, num_valid_seq=1)

#%%
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
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.h_dim),
                torch.zeros(self.n_layers, batch_size, self.h_dim))
    
class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=612, n_layers=4, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
       
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden



#%%

train_ids = text_to_ids(text=train_text, token_types=token_types)
train_encoded = ids_to_encoded(ids=train_ids, token_types=token_types)

valid_ids = text_to_ids(text=valid_text, token_types=token_types)
valid_encoded = ids_to_encoded(ids=valid_ids, token_types=token_types)

print("train")
print(train_encoded.shape)

print("valid")
print(valid_encoded.shape)

train_x, train_y = prepare_sequences(train_encoded, seq_len=30)
valid_x, valid_y = prepare_sequences(valid_encoded, seq_len=30)

train_y = encoded_to_ids(train_y, batched=True)
valid_y = encoded_to_ids(valid_y, batched=True)

#%%

model = CharRNN(token_types, 20, 1, 0.0)
#model = LSTMModel(len(token_types), 20, n_layers=1)
initial_hidden = model.init_hidden(1)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%%

def train(model, n_epochs, train_set, valid_set, loss_fn, optimizer, clip=5):
    model.train()
    losses = []
    
    for epoch in range(0, n_epochs):
        avg_epoch_loss = 0
        hidden = model.init_hidden(1)
        for i in range(train_set[0].shape[0]):
            #print(encoded_to_ids(train_set[0][i], False))
            #print(train_set[1][i])
            
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
            
            avg_epoch_loss += loss.item() / train_set[0].shape[0]
        
        losses.append(avg_epoch_loss)
        print("\t Epoch {}/{}: train loss: {}".format(epoch + 1, n_epochs, avg_epoch_loss))
    
    return losses

losses = train(model, 10, (train_x, train_y), (valid_x, valid_y), loss_fn, optimizer, clip=5)

plt.plot(losses)


#%%


def generate(model, initial_text, seq_len):
    model.eval()
    
    sequence = torch.tensor(np.array([text_to_encoded(initial_text, token_types)])).float()
    hidden = model.init_hidden(1)
    
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
    return initial_text + encoded_to_text(np.squeeze(np.array(generated), axis=1), token_types)

print(generate(model, "a", 100))





