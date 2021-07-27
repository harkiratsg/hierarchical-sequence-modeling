import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CharLSTM(nn.Module):
    def __init__(self, n_tokens, n_hidden, n_layers=1, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.lstm = nn.LSTM(n_tokens, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, n_tokens)
      
    
    def forward(self, x, hidden): 
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.n_hidden)
        y = self.fc(out)
        return y, hidden
    
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden),
                torch.zeros(self.n_layers, batch_size, self.n_hidden))
    
    

class DenseCharLSTM(nn.Module):
    def __init__(self, n_tokens, n_hidden, n_layers=1, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.n_tokens = n_tokens
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.lstm = nn.LSTM(n_tokens, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, n_tokens)
        
        self.fc_res = nn.Linear(n_tokens, n_tokens)
      
    
    def forward(self, x, hidden):
        res = self.fc_res(x).contiguous().view(-1, self.n_tokens)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.n_hidden)
        y = self.fc(out) + res
        return y, hidden
    
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden),
                torch.zeros(self.n_layers, batch_size, self.n_hidden))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        #print("encoded:", src.shape)
        src = self.pos_encoder(src)
        #print("pose + encoded:",src.shape)
        output = self.transformer_encoder(src, src_mask)
        #print("transformed:", output.shape)
        output = self.decoder(output)
        #print("decoded:", output.shape)
        return output
    
