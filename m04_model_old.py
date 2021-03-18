# MODEL_02 - NN CLASS

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# artificial neural network 
class ANN(nn.Module):
    def __init__(self, cfg):
        super(ANN, self).__init__()
        
        self.cfg = cfg
        
        # lstm layer with the swl in the inout --> input_size = 15
        self.lstm_swl = nn.LSTM(self.cfg.input_size_swl, self.cfg.hidden_size,
                          self.cfg.num_lstm_layers, batch_first = True)
        
        # lstm layer without the swl in the inout --> input_size = 14
        self.lstm = nn.LSTM(self.cfg.input_size, self.cfg.hidden_size,
                      self.cfg.num_lstm_layers, batch_first = True)
        
        # readout layer
        self.fc1 = nn.Linear(self.cfg.hidden_size * self.cfg.output_size, cfg.output_size)
        # self.fc1 = nn.Linear(self.cfg.hidden_size, 1)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
        self.Sigmoid = nn.Sigmoid()
        
        self.fc2 = nn.Linear(1, self.cfg.hidden_size)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)

    def forward(self, input_swl, input):
        # set initial hidden and cell state
        # h_0 = torch.zeros(self.cfg.num_lstm_layers, input.size(0), self.cfg.hidden_size).to(self.cfg.device)
        
        # inp_swl = input_swl[:,0,:1]
        # swl_encode = self.fc2(inp_swl)
        # swl_encode = swl_encode.unsqueeze(0)
        # swl_encode = self.Sigmoid(swl_encode)
        
        swl_encode = self.fc2(input_swl.squeeze(1)).unsqueeze(0)
        swl_encode = self.Sigmoid(swl_encode)
        
        h_0 = torch.cat((swl_encode.to(self.cfg.device),torch.zeros(self.cfg.num_lstm_layers-1,
                        input.size(0), self.cfg.hidden_size).to(self.cfg.device))).to(self.cfg.device)
        
        c_0 = torch.zeros(self.cfg.num_lstm_layers, input.size(0),
                          self.cfg.hidden_size).to(self.cfg.device)

        # forward propagate
        # out_swl, (hn_swl, cn_swl) = self.lstm_swl(input_swl, (h_0, c_0))
        # out_lstm, (hn, cn) = self.lstm(input, (hn_swl, cn_swl))
        # cat_inp = torch.cat((input_swl[:,0,1:].unsqueeze(1).to(self.cfg.device),
        #                       input),dim=1).to(self.cfg.device)
        
        # out_lstm, (hn, cn) = self.lstm(cat_inp, (h_0, c_0))
        # out = torch.cat((out_swl, out_lstm), 1)
        # out = out.reshape(input.size(0), 1, self.cfg.hidden_size * self.cfg.output_size)
        
        out_lstm, (hn, cn) = self.lstm(input, (h_0, c_0))
        out = out_lstm.reshape(input.size(0), 1, self.cfg.hidden_size * self.cfg.output_size)
        #out = out.permute()
        # out = self.fc1(out_lstm)
        out = self.fc1(out)
        # out = out.permute(0,2,1)
        out = self.Sigmoid(out)
        
        return out