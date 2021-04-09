"""
M04_MODEL.PY
This script contains the ANN class that constructs the architecture of the
LSTM model
"""

import torch


class ANN(torch.nn.Module):
    """
    This class initialize data loading from the .csv file and pre-process
    (normalization and get statistics of data)
    """
    
    def __init__(self, cfg):
        """
        Constructor
        """
        
        super(ANN, self).__init__()
        
        self.cfg = cfg
        
        # Initialize the lstm cell with the arguments:
        #   input_size: the feature size of the input tensor
        #   hidden_size: the feature size of the hidden states
        #   num_lstm_layers: depth of the lstm cells (number of lstm layers)
        
        # Because batch_first = True, then the input to the LSTM will be with
        # dimensions: [batch_size, seq_len, feature_size]
        self.lstm = torch.nn.LSTM(self.cfg.input_size, self.cfg.hidden_size,
                      self.cfg.num_lstm_layers, batch_first = True)
        
        # Initialize the encoder layer to encode the initial SWL value into
        # the LSTM hidden state in the first layer
        #   in_features = 1: number of input features, in this case is 1 because
        #       the inputs_swl tensor array only contains 1 SWL value in each
        #       batch/sample
        #   out_features = hidden_size: number of output features, in this case
        #       is the feature size in the LSTM hidden states
        #   In general, batch_size is not necessary to be defined when using
        #       torch.nn.Linear, it handles the batch size automatically
        self.encoder = torch.nn.Linear(1, self.cfg.hidden_size)
        torch.nn.init.normal_(self.encoder.bias, mean=0.0, std=1.0)
        
        # Initialize the decoder layer to decode the LSTM hidden state output
        # into the space of SWL variable
        #   in_features = hidden_size*output_size: number of input features,
        #       in this case is the feature size of the hidden state multiplied
        #       by the output size (= forecast_horizon)
        #   out_features = output_size: number of output features, in this case
        #       is the forecast horizon to store all the SWL values in the
        #       corresponding horizon
        self.decoder = torch.nn.Linear(self.cfg.hidden_size * self.cfg.output_size, cfg.output_size)
        torch.nn.init.normal_(self.decoder.bias, mean=0.0, std=1.0)



    def forward(self, input_swl, input):
        """
        This function defines each forward pass of the LSTM model
        """
       
        # Feed the input_swl array into the encoder
        # input_swl is first squeezed along the first dimension, because its
        # seq_len = 1 to fit the encoder input dimension requirement
        
        # The output of the encoder is then unsqueezed along the 0-th dimension
        # to be concatenated later as part of the initial hidden state h_0
        # because the 0-th dimension in the hidden state shape is the number of
        # layers and not batch size
        swl_encode = self.encoder(input_swl.squeeze(1)).unsqueeze(0)
        swl_encode = torch.sigmoid(swl_encode)
        
        # Concatenate the output of the encoder into the initial hidden state
        # in the first layer of the LSTM model
        
        # The hidden state and cell state have a shape of [num_layers, batch_size, hidden_size]
        # Therefore, the output of the encoder is concatenated with a zero
        # array with the shape of [num_layers-1, batch_size, hidden_size] along
        # the 0-th dimension
        
        h_0 = torch.cat((swl_encode.to(self.cfg.device),torch.zeros(self.cfg.num_lstm_layers-1,
                        input.size(0), self.cfg.hidden_size).to(self.cfg.device))).to(self.cfg.device)
        
        # Initialize a zero array for the initial cell state
        c_0 = torch.zeros(self.cfg.num_lstm_layers, input.size(0),
                          self.cfg.hidden_size).to(self.cfg.device)

        # Once initialized, feed the input along with the initial hidden and cell
        # state into the LSTM model
        out_lstm, (hn, cn) = self.lstm(input, (h_0, c_0))
        
        # Reshape the output of the LSTM model to comply with the decoder dimension
        # requirement: [batch_size, *, in_size]
        # Here, in_size = hidden_size*output_size because the decoder is a fully
        # connected layer
        out = out_lstm.reshape(input.size(0), 1, self.cfg.hidden_size * self.cfg.output_size)

        # Feed the LSTM output into the decoder, and permute to reshape according
        # to the shape [batch_size, seq_len, feature_size]
        out = self.decoder(out)
        out = out.permute(0,2,1)
        out = torch.sigmoid(out)
        
        return out