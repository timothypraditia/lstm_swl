"""
M02_INITIALIZATION.PY
This script constructs an object with all the configuration required
"""


import torch
import os
import shutil
import m00_configuration as params

class Initialize:
    """
    This class holds the configuration for the model
    """

    def __init__(self):
        
        # SET WORKING PATH
        self.main_path = os.getcwd()
        
        
        # DATA CONFIGURATION
        self.data_filename = params.data_filename
        self.data_path = self.main_path + '\\' + self.data_filename
        self.normalize = params.normalize
        self.percentage_train = params.percentage_train
        self.percentage_vali = params.percentage_vali
        self.percentage_test = params.percentage_test
        #Input size and prediction horizon:
        self.in_swl = params.in_swl
        self.in_p = params.in_p
        self.in_et = params.in_et
        self.in_w5 = params.in_w5
        self.in_w6 = params.in_w6
        self.in_p_forecast = params.in_p_forecast
        self.in_et_forecast = params.in_et_forecast
        self.in_w_forecast = params.in_w_forecast
        self.forecast_horizon = params.forecast_horizon
        self.input_size = params.input_size
        self.output_size = params.output_size
        
        
        # MODEL NAME & SETTING
        self.model_name = params.model_name
        self.save_model = params.save_model
        self.continue_training = params.continue_training
        self.device_name = params.device_name
        self.device = self.determine_device()
        self.model_path = self.main_path + '\\' + self.model_name
        self.check_dir(self.model_path)
        self.log_path = self.main_path + '\\runs\\' + self.model_name
        # Remove old log files to prevent unclear visualization in tensorboard
        self.check_dir(self.log_path, remove=True)
        
        # NETWORK HYPER-PARAMETERS
        self.num_lstm_layers = params.num_lstm_layers
        self.hidden_size = params.hidden_size
        self.learning_rate = params.learning_rate
        self.phys_mult = params.phys_mult
        self.weight_decay = params.weight_decay
        self.epochs = params.epochs
        self.lbfgs_optim = params.lbfgs_optim
        self.minibatch = params.minibatch
        if self.lbfgs_optim:
            self.minibatch = False
        self.physics_constraint = params.physics_constraint
        self.num_samples_phys = params.num_samples_phys
        if self.device_name == 'cuda':
            self.num_samples_phys = 500 #To avoid insufficient memory in GPU application
        
        
        
    def determine_device(self):
        """
        This function evaluates whether a GPU is accessible at the system and
        returns it as device to calculate on, otherwise it returns the CPU.
        :return: The device where tensor calculations shall be made on
        """
        
        self.device = torch.device(self.device_name)
        print("Using device:", self.device)
        print()
        
        # Additional Info when using cuda
        if self.device.type == "cuda" and torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print("\tAllocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
            print("\tCached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
            print()
        
        return self.device
    
    
    
    def check_dir(self, path_dir, remove=False):
        """
        This function evaluates whether a directory for the corresponding model
        exists, otherwise create a new directory
        """
        
        # For tensorboard log files, clean the directory to avoid error in
        # visualization
        if remove:
            if os.path.exists(path_dir): # check whether the specified path exists or not
                shutil.rmtree(path_dir)
            
        # If the path does not exist, create a new path
        if not os.path.exists(path_dir):
            os.makedirs(path_dir) # create a directory
    