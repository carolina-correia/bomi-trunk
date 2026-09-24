#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_dim, pre_output_size, output_size, n_layers, dropout):
        super(LSTM, self).__init__()

        # internal params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pre_output_size = pre_output_size 

        # network
        self.lstm = nn.LSTM(input_size, 
                            hidden_dim, 
                            n_layers,
                            batch_first=True, 
                            dropout=dropout, 
                            proj_size=0) # LSTM hidden units
        
        if self.pre_output_size == 0:
            self.fc = nn.Linear(hidden_dim, output_size) # output layer
        else:
            self.fc = nn.Linear(hidden_dim, pre_output_size) # output layer
            self.fc_2 = nn.Linear(pre_output_size, output_size) # output layer
        
    def forward(self, x, h0=None, c0=None):
        # batch size
        batch_size = x.shape[0]
        # hidden state and cell state
        if h0 is None and c0 is None:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros_like(h0)

        hidden = (c0, h0)
        out, hidden = self.lstm(x, hidden)
        h,c = hidden

        out = out.view(batch_size, -1, self.hidden_dim)
        out = self.fc(out[:, -1, :])
        
        if self.pre_output_size != 0:
            out = F.relu(out)
            out = self.fc_2(out)

        return out, (h, c)