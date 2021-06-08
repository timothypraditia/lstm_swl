"""
M03_PRE_PROCESS.PY
This script contains classes necessary for the data initialization, pre-
processing, and setup of dataloader
"""

import pandas as pd
import torch
import copy
from torch.utils.data import Dataset

class Initialize_Data:
    """
    This class initialize data loading from the .csv file and pre-process
    (normalization and get statistics of data)
    """
    
    def __init__(self, cfg):
        """
        Constructor
        """
        
        # Read data from a .csv file
        self.data = pd.read_csv(cfg.data_path, sep = '\t')
        
        # Transform dataframe to numpy arrays
        self.inp_swl = self.data['Grundwasserstand  [m ü. NN]'].to_numpy()
        self.inp_p = self.data['Niederschlag  [mm/h]'].to_numpy()
        self.inp_t = self.data['Lufttemperatur  [°C]'].to_numpy()
        self.inp_sd = self.data['Sonnenscheindauer  [min/h]'].to_numpy()
        self.inp_rh = self.data['relative Luftfeuchte  [%]'].to_numpy()
        self.inp_wv = self.data['Windgeschwindigkeit  [m/s]'].to_numpy()   
        self.inp_w5 = self.data['Entnahmerate_w_5  [l/s]'].to_numpy()
        self.inp_w6 = self.data['Entnahmerate_w_6  [l/s]'].to_numpy()
        self.tar_swl = self.data['Grundwasserstand  [m ü. NN]'].to_numpy()
        
        # Convert the numpy arrays to tensor arrays
        self.inp_swl = torch.FloatTensor(self.inp_swl)
        self.inp_p = torch.FloatTensor(self.inp_p)
        self.inp_t = torch.FloatTensor(self.inp_t)
        self.inp_sd = torch.FloatTensor(self.inp_sd)
        self.inp_rh = torch.FloatTensor(self.inp_rh)
        self.inp_wv = torch.FloatTensor(self.inp_wv)
        self.inp_w5 = torch.FloatTensor(self.inp_w5)
        self.inp_w6 = torch.FloatTensor(self.inp_w6)
        self.tar_swl = torch.FloatTensor(self.tar_swl)
        
        # Reshape the tensor arrays
        # The tensor arrays will have the dimension of [sample_size,1]
        self.inp_swl = self.inp_swl.view(-1,1)
        self.inp_p = self.inp_p.view(-1,1)
        self.inp_t = self.inp_t.view(-1,1)
        self.inp_sd = self.inp_sd.view(-1,1)
        self.inp_rh = self.inp_rh.view(-1,1)
        self.inp_wv = self.inp_wv.view(-1,1)
        self.inp_w5 = self.inp_w5.view(-1,1)
        self.inp_w6 = self.inp_w6.view(-1,1)
        self.tar_swl = self.tar_swl.view(-1,1)
        
        # Get data statistics and normalize if necessary
        self.get_data_stats(cfg)
        
        if cfg.normalize:
            self.normalize_data(cfg)
        
        # Calculate number of samples to be divided into training, validation, and testing sets
        cfg.count_train = round(len(self.data) * cfg.percentage_train)
        cfg.count_vali = round(len(self.data)* cfg.percentage_vali)
        cfg.count_test = round(len(self.data)* cfg.percentage_test)
        
        if not cfg.count_train + cfg.count_vali + cfg.count_test == len(self.inp_swl):
            cfg.count_test = len(self.inp_swl) - cfg. count_train - cfg.count_vali
        
        # The batch size for the training dataset is adjusted to compensate for
        # the first forecast horizon (first target is calculated at time step
        # t = forecast_horizon) and 1 time step that will be used as the forecast
        # input
        cfg.batch_size_train = cfg.count_train - max(cfg.in_p, cfg.in_t, cfg.in_sd, cfg.in_rh, cfg.in_wv, cfg.in_swl) \
                            - cfg.forecast_horizon
        cfg.batch_size_vali = cfg.count_vali
        cfg.batch_size_test = cfg.count_test
    
    
    
    def get_data_stats(self, cfg):
        """
        This function retrieves the data statistics required for normalization
        """
        
        # Find minimum value of each variable
        cfg.min_swl = torch.min(self.inp_swl) - 0.15 # Decreasing by 15 cm
        cfg.min_p = torch.min(self.inp_p) # Minimum value is already 0 
        cfg.min_t = torch.min(self.inp_t) * 1.15 # Decreasing by 15 percent
        cfg.min_sd = torch.min(self.inp_sd) # Minimum value is already 0 
        cfg.min_rh = torch.min(self.inp_rh) * 0.85 # Decreasing by 15 percent
        cfg.min_wv = torch.min(self.inp_wv) # Minimum value is already 0 
        cfg.min_w5 = torch.min(self.inp_w5) # Minimum value is already 0 
        cfg.min_w6 = torch.min(self.inp_w6) # Minimum value is already 0 
        
        # Find maximum value of each variable
        cfg.max_swl = torch.max(self.inp_swl) + 0.15 # Increasing by 15 cm
        cfg.max_p = torch.max(self.inp_p) * 1.15 # Increasing by 15 percent
        cfg.max_t = torch.max(self.inp_t) * 1.15 # Increasing by 15 percent
        cfg.max_sd = torch.max(self.inp_sd) # Maximum possible value already included
        cfg.max_rh = torch.max(self.inp_rh) # Maximum possible value already included
        cfg.max_wv = torch.max(self.inp_wv) * 1.15 # Increasing by 15 percent
        cfg.max_w5 = torch.max(self.inp_w5) # Maximum possible value already included
        cfg.max_w6 = torch.max(self.inp_w6) # Maximum possible value already included
        
        
        
    def normalize_data(self, cfg):
        """
        This function normalize the data into the range [0,1] using the maximum
        and minimum value
        """
        
        self.inp_swl = (self.inp_swl - cfg.min_swl)/(cfg.max_swl-cfg.min_swl)
        self.inp_p = (self.inp_p - cfg.min_p)/(cfg.max_p-cfg.min_p)
        self.inp_t = (self.inp_t - cfg.min_t)/(cfg.max_t - cfg.min_t)
        self.inp_sd = (self.inp_sd - cfg.min_sd)/(cfg.max_sd - cfg.min_sd)
        self.inp_rh = (self.inp_rh - cfg.min_rh)/(cfg.max_rh - cfg.min_rh)
        self.inp_wv = (self.inp_wv - cfg.min_wv)/(cfg.max_wv - cfg.min_wv)
        self.inp_w5 = (self.inp_w5 - cfg.min_w5)/(cfg.max_w5-cfg.min_w5)
        self.inp_w6 = (self.inp_w6 - cfg.min_w6)/(cfg.max_w6-cfg.min_w6)
        self.tar_swl = (self.tar_swl - cfg.min_swl)/(cfg.max_swl-cfg.min_swl)
        



class Load_Data(Dataset):
    """
    This class setup the data loader by first reshaping the input tensors and
    then defining the __len__ and __getitem__ function for the data loader
    enumerator
    """
    
    def __init__(self, dataset, cfg, training=False, validation=False, testing=False):
        """
        Constructor
        """
        
        # Ensure that every tensor has the same length
        assert len(dataset.inp_swl) == len(dataset.tar_swl)
        assert len(dataset.inp_p) == len(dataset.tar_swl)
        assert len(dataset.inp_t) == len(dataset.tar_swl)
        assert len(dataset.inp_sd) == len(dataset.tar_swl)
        assert len(dataset.inp_rh) == len(dataset.tar_swl)
        assert len(dataset.inp_wv) == len(dataset.tar_swl)
        assert len(dataset.inp_w5) == len(dataset.tar_swl)
        assert len(dataset.inp_w6) == len(dataset.tar_swl)
        
        self.cfg = cfg
        
        # Calculate indices required for data slicing into training, validation, and testing
        if training:
            self.batch_size = cfg.batch_size_train
            self.count_start = 0
            self.count_end = cfg.count_train
        elif validation:
            self.batch_size = cfg.batch_size_vali
            self.count_start = cfg.count_train - cfg.forecast_horizon - 1
            self.count_end = cfg.count_train + cfg.count_vali
        elif testing:
            self.batch_size = cfg.batch_size_test
            self.count_start = cfg.count_train + cfg.count_vali - cfg.forecast_horizon - 1
            self.count_end = cfg.count_train + cfg.count_vali + cfg.count_test
        
        
        # Create a copy of the dataset to avoid changing the original values
        # as well as differentiating between training, validation, and testing data
        self.create_copy(dataset)
        
        # Slice data into training, validation and testing sets        
        self.inp_swl = self.inp_swl[self.count_start : self.count_end]
        self.inp_p = self.inp_p[self.count_start : self.count_end]
        self.inp_t = self.inp_t[self.count_start : self.count_end]
        self.inp_sd = self.inp_sd[self.count_start : self.count_end]
        self.inp_rh = self.inp_rh[self.count_start : self.count_end]
        self.inp_wv = self.inp_wv[self.count_start : self.count_end]
        self.inp_w5 = self.inp_w5[self.count_start : self.count_end]
        self.inp_w6 = self.inp_w6[self.count_start : self.count_end]
        self.tar_swl = self.tar_swl[self.count_start : self.count_end]
        
        self.reshape_data()
        
        
        
    def create_copy(self, dataset):
        """
        This function creates a deepcopy of the tensor arrays to avoid modifying
        the original arrays
        """
        
        self.inp_swl = copy.deepcopy(dataset.inp_swl) #deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original
        self.inp_p = copy.deepcopy(dataset.inp_p)
        self.inp_t = copy.deepcopy(dataset.inp_t)
        self.inp_sd = copy.deepcopy(dataset.inp_sd)
        self.inp_rh = copy.deepcopy(dataset.inp_rh)
        self.inp_wv = copy.deepcopy(dataset.inp_wv)
        self.inp_w5 = copy.deepcopy(dataset.inp_w5)
        self.inp_w6 = copy.deepcopy(dataset.inp_w6)
        self.tar_swl = copy.deepcopy(dataset.tar_swl)
        
        
    
    def reshape_data(self):
        """
        This function reshapes the input and target tensor for compatibility to
        the LSTM model
        This function is useful to avoid performing iterations inside the
        __getitem__ function, which will cause huge overhead during training
        """
        
        # In general, the arrays have dimensions of [batch_size, seq_len, feature_size]
        # to comply with the LSTM dimension handle
        # Initialize the tensor arrays:
        #   inputs_swl: dim[batch_size, 1, 1]
        #       contains the SWL values at only the initial time step as
        #       initial input to the LSTM
        #   inputs: dim[batch_size, forecast_horizon, input_size]
        #       contains the input values with the order of: precipitation,
        #       precipitation forecast, air temperature, air temperature
        #       forecast, sunshine duration, sunshine duration forecast, relative humidity,
        #       relative humidity forecast, wind velocity, wind velocity forecast,
        #       well 5 pumping rate, well 5 pumping rate forecast,
        #       well 6 pumping rate, well 6 pumping rate forecast
        #   target: dim[batch_size, forecast_horizon, 1]
        #       contains the target SWL values until time step t = t_init + forecast_horizon
        
        self.inputs_swl = torch.zeros(self.batch_size, 1, 1)
        self.inputs = torch.zeros(self.batch_size,self.cfg.forecast_horizon, self.cfg.input_size)
        self.target = torch.zeros(self.batch_size, self.cfg.forecast_horizon, 1)
        
        for i in range(self.batch_size):
            # Assign values to the inputs_swl tensor array using data from
            # tensor inp_swl at the same corresponding batch/sample index
            self.inputs_swl[i,0] = self.inp_swl[i]
            
            # Assign values to the inputs tensor array using data from tensors
            # inp_p, inp_et, inp_w5, and inp_w6, each at the corresponding batch/
            # sample index, and also the forecast at index + 1
            
            # The time steps covered range from t0 = 0 to t_end = t0 + forecast horizon
            for t in range(self.cfg.forecast_horizon):
                self.inputs[i,t] = torch.cat(
                (self.inp_p[i + t : i + self.cfg.in_p + self.cfg.in_p_forecast + t],
                 self.inp_t[i + t : i + self.cfg.in_t + self.cfg.in_t_forecast + t],
                 self.inp_sd[i + t : i + self.cfg.in_sd + self.cfg.in_sd_forecast + t],
                 self.inp_rh[i + t : i + self.cfg.in_rh + self.cfg.in_rh_forecast + t],
                 self.inp_wv[i + t : i + self.cfg.in_wv + self.cfg.in_wv_forecast + t],
                 self.inp_w5[i + t : i + self.cfg.in_w5 + self.cfg.in_w_forecast + t],
                 self.inp_w6[i + t : i + self.cfg.in_w6 + self.cfg.in_w_forecast + t])).squeeze()
            
            # Assign values to the target tensor array using data from tensor
            # inp_swl, offset by 1 time step
            self.target[i] = self.tar_swl[i + 1 : i + 1 + self.cfg.forecast_horizon]
            
            
        
    def __len__(self):
        """
        This function returns the sample length (batch size) of each dataset
        """
        
        return self.batch_size
    

    
    def __getitem__(self, idx):
        """
        This function returns the sample pointed by the idx integer
        """
        return self.inputs_swl[idx], self.inputs[idx], self.target[idx]