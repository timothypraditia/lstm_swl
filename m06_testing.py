"""
M06_TESTING.PY
This script contains the testing class that is used to evaluate the trained
LSTM model
"""

import torch
import torch.nn as nn


class Testing:
    
    def __init__(self, cfg, trainer):
        """
        Constructor
        """
        
        self.cfg = cfg
        
        self.trainer = trainer
        
        
    def evaluate(self, test_loader):
        """
        This function evaluates the trained model with the test dataset
        """
        
        # Obtain the whole testing dataset
        inputs_swl_test, inputs_test, target_test = next(iter(test_loader))
        
        # For testing, the torch.no_grad() wrapper is needed so that the
        # forward propagation is not tracked, because backpropagation is not
        # needed
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.trainer.model.eval()
            
            # Send the inputs and target to the corresponding device
            inputs_swl_test = inputs_swl_test.to(self.cfg.device)
            inputs_test = inputs_test.to(self.cfg.device)
            target_test = target_test.to(self.cfg.device)
            
            # Calculate the prediction using the testing dataset
            lstm_output_test = self.trainer.model(inputs_swl_test, inputs_test)
    
            # Calculate the testing loss
            lstm_test_loss = self.trainer.criterion(lstm_output_test, target_test)
            self.test_loss = lstm_test_loss.item()
            
            print()
            print('Testing Loss: %.4f' %(self.test_loss))
            print()
            
            # Sample random indices for the physics constraint calculation
            # If full batch is used, memory requirement and computation time
            # would be too high
            self.rand_idx = torch.randint(0, self.cfg.batch_size_test,
                                     (self.cfg.num_samples_phys,1)).squeeze()
            
            # Calculate the physical error
            
            # Precipitation
            # Modify the precipitation values to 0.50, 0.75, 1.00, 1.25, and 1.50
            # times the original values
            inp_temp_p = []
            self.lstm_output_test_p = []
            self.lstm_error_p = 0
            
            for i in range(5):
                inp_temp_p.append(inputs_test[self.rand_idx].clone().detach())
                
                # Modify the precipitation values
                inp_temp_p[i][:,:,0:2] = inputs_test[self.rand_idx,:,0:2].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                self.lstm_output_test_p.append(self.trainer.model(inputs_swl_test[self.rand_idx],
                                                inp_temp_p[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the precipitation values, the SWL prediction should
                # be higher
                if i >= 1:
                    self.lstm_error_p += torch.mean(torch.relu(
                        self.lstm_output_test_p[i-1] - self.lstm_output_test_p[i])) / 5
            
            print('Precipitation physical error = %.2e' %(self.lstm_error_p.item()))
            
            
            # Evapotranspiration
            # Modify the evapotranspiration values to 0.50, 0.75, 1.00, 1.25, and 1.50
            # times the original values
            inp_temp_et = []
            self.lstm_output_test_et = []
            self.lstm_error_et = 0
            
            for i in range(5):
                inp_temp_et.append(inputs_test[self.rand_idx].clone().detach())
                
                # Modify the evapotranspiration values
                inp_temp_et[i][:,:,2:4] = inputs_test[self.rand_idx,:,2:4].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                self.lstm_output_test_et.append(self.trainer.model(inputs_swl_test[self.rand_idx],
                                                inp_temp_et[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the evapotranspiration values, the SWL prediction
                # should be lower
                if i >= 1:
                    self.lstm_error_et += torch.mean(torch.relu(
                        self.lstm_output_test_et[i] - self.lstm_output_test_et[i-1])) / 5
            
            print('Evapotranspiration physical error = %.2e' %(self.lstm_error_et.item()))
                    
            
            # Well 5
            # Modify the well 5 pumping rate values to 0.00, 0.25, 0.50, 0.75,
            # and 1.00 (absolute values) for better comparison visualization
            inp_temp_w5 = []
            self.lstm_output_test_w5 = []
            self.lstm_error_w5 = 0
            
            for i in range(5):
                inp_temp_w5.append(inputs_test[self.rand_idx].clone().detach())
                
                # Modify the well 5 pumping rate values
                inp_temp_w5[i][:,:,4:6] = torch.zeros_like(inputs_test[self.rand_idx,:,4:6]) + 0.25*i
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                self.lstm_output_test_w5.append(self.trainer.model(inputs_swl_test[self.rand_idx],
                                                inp_temp_w5[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the well 5 pumping rate values, the SWL prediction
                # should be lower
                if i >= 1:
                    self.lstm_error_w5 += torch.mean(torch.relu(
                        self.lstm_output_test_w5[i] - self.lstm_output_test_w5[i-1])) / 5
            
            print('Well 5 pumping rate physical error = %.2e' %(self.lstm_error_w5.item()))
            
            # Well 6
            # Modify the well 6 pumping rate values to 0.00, 0.25, 0.50, 0.75,
            # and 1.00 (absolute values) for better comparison visualization
            inp_temp_w6 = []
            self.lstm_output_test_w6 = []
            self.lstm_error_w6 = 0
            
            for i in range(5):
                inp_temp_w6.append(inputs_test[self.rand_idx].clone().detach())
                
                # Modify the well 6 pumping rate values
                inp_temp_w6[i][:,:,6:] = torch.zeros_like(inputs_test[self.rand_idx,:,6:]) + 0.25*i
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                self.lstm_output_test_w6.append(self.trainer.model(inputs_swl_test[self.rand_idx],
                                                inp_temp_w6[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the well 6 pumping rate values, the SWL prediction
                # should be lower
                if i >= 1:
                    self.lstm_error_w6 += torch.mean(torch.relu(
                        self.lstm_output_test_w6[i] - self.lstm_output_test_w6[i-1])) / 5
            
            print('Well 6 pumping rate physical error = %.2e' %(self.lstm_error_w6.item()))
            print()
        
        # Update the configuration and performance log file
        self.update_config_log()
        
        # Write the hyperparameters values to the tensorboard log file
        self.write_hparams()
        
        
    
    def update_config_log(self):
        """
        This function updates the configuration and performance log file
        """
        
        # Open the configuration and performance log file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'r') as f:
            cfg_file = f.read()
            
        output_string = cfg_file + "\n#\n# Testing Performance\n\n"
    
        output_string += "TESTING_ERROR = " + \
                         str(self.test_loss) + "\n"
        output_string += "PHYSICAL_ERROR_PRECIPITATION = " + \
                         str(self.lstm_error_p.item()) + "\n"
        output_string += "PHYSICAL_ERROR_EVAPOTRANSPIRATION = " + \
                         str(self.lstm_error_et.item()) + "\n"
        output_string += "PHYSICAL_ERROR_WELL_5 = " + \
                         str(self.lstm_error_w5.item()) + "\n"
        output_string += "PHYSICAL_ERROR_WELL_6 = " + \
                         str(self.lstm_error_w6.item()) + "\n"
    
        # Save the updated performance metrics into the file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
            _text_file.write(output_string)
    
    
    
    def write_hparams(self):
        """
        This function writes the hyperparameters values to the tensorboard
        log file
        """
        
        # Extract the training and validation loss values from the latest
        # checkpoint (i.e. with the best performance)
        train_loss_save = self.trainer.checkpoint['loss_train'][-1]
        vali_loss_save = self.trainer.checkpoint['loss_vali'][-1]
        
        # Define the hparam_dict that contains the hyperparameters values
        hparam_dict = {'num_layers': self.cfg.num_lstm_layers,
                       'hidden_size': self.cfg.hidden_size,
                       'lr': self.cfg.learning_rate,
                       'phys_mult': self.cfg.phys_mult,
                       'weight_decay': self.cfg.weight_decay,
                       'num_epochs': self.cfg.epochs,
                       'lbfgs': self.cfg.lbfgs_optim,
                       'minibatch': self.cfg.minibatch,
                       'phys_const': self.cfg.physics_constraint,
                       'num_phys_samples': self.cfg.num_samples_phys}
        
        # Define the metric_dict that contains the performance metrics values
        metric_dict = {'train_loss': train_loss_save,
                       'vali_loss': vali_loss_save,
                       'test_loss': self.test_loss,
                       'phys_loss_p': self.lstm_error_p.item(),
                       'phys_loss_et': self.lstm_error_et.item(),
                       'phys_loss_w5': self.lstm_error_w5.item(),
                       'phys_loss_w6': self.lstm_error_w6.item()
                       }
        
        # Write the hparams_dict and metric_dict to the tensorboard log file
        self.trainer.tb.add_hparams(hparam_dict, metric_dict)