"""
M05_TRAINING.PY
This script contains the training class that is used to train the LSTM model
"""

import torch
import torch.nn as nn
import numpy as np
import time
from threading import Thread
from torch.utils.tensorboard import SummaryWriter


class Training:
    
    def __init__(self, cfg, model):
        """
        Constructor
        """
        
        self.cfg = cfg
        
        # Send model to the corresponding device (important when using GPU)
        self.model = model.to(self.cfg.device)
        
        # Use MSE as the loss function definition
        self.criterion = nn.MSELoss()
        
        # Choose between ADAM or LBFGS as the optimizer
        # LBFGS theoretically should work better compared to ADAM, but the
        # memory requirement and computation time is also higher
        
        if self.cfg.lbfgs_optim:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr = self.cfg.learning_rate)
        else:
            # The weight decay input determines the multiplier for the L2 normalization
            self.optimizer = torch.optim.Adam(model.parameters(), lr = self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        
        
        self.start_epoch = 0
        self.train_loss = []
        self.vali_loss = []
        self.best_vali = np.infty
        
        # Define the filename to save and/or load the model
        self.model_save_file = self.cfg.model_path + "\\" + self.cfg.model_name + ".pt"
        
        # Create a Tensorboard summary writer instance in the log directory
        # The Tensorboard summary includes the training and validation loss,
        # as well as hyperparameters values to be compared with other models
        self.tb = SummaryWriter(self.cfg.log_path)
        
        
        # Load the model if this instance is a training continuation from a
        # previous checkpoint
        if self.cfg.continue_training:
            print('Restoring model (that is the network\'s weights) from file...')
            print()
            
            # Load the latest checkpoint
            self.checkpoint = torch.load(self.model_save_file)
            
            # Load the model state_dict (all the network parameters) and send
            # the model to the corresponding device
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.model.to(self.cfg.device)
            
            # Load the optimizer state dict (important because ADAM and LBFGS 
            # requires past states, e.g. momentum information and approximate
            # Hessian)
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.cfg.device)
            
            # Load the epoch and loss values from the previous training up until
            # the checkpoint to enable complete history of the training
            self.start_epoch = self.checkpoint['epoch']
            self.train_loss = self.checkpoint['loss_train']
            self.vali_loss = self.checkpoint['loss_vali']
            
            # Store the loss values in the Tensorboard log file
            for epoch in range(self.start_epoch):
                self.tb.add_scalar('training_loss', self.train_loss[epoch], epoch)
                self.tb.add_scalar('validation_loss', self.vali_loss[epoch], epoch)
            
            
            
    def train(self, train_loader, vali_loader):
        """
        This is the main function for the training
        """
        
        # Train with mini batch or full batch
        # LBFGS works only with full batch, but the parameter cfg.minibatch
        # should have been adjusted in the initialization script
        if self.cfg.minibatch:
            self.minibatch_training(train_loader, vali_loader)
        else:
            self.batch_training(train_loader, vali_loader)
        
        
        # Load model from the latest saved checkpoint (i.e. with the lowest
        # validation error)
        self.checkpoint = torch.load(self.model_save_file)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.to(self.cfg.device)
        
        
        
        
    def batch_training(self, train_loader, vali_loader):
        """
        This function trains the model with the full batch method
        """
        # Set the number of threads for this program to one
        torch.set_num_threads(1)
        
        # Obtain the whole training and validation dataset for the full batch
        # training
        inputs_swl_train, inputs_train, target_train = next(iter(train_loader))
        inputs_swl_vali, inputs_vali, target_vali = next(iter(vali_loader))
        
        for epoch in range (self.start_epoch, self.cfg.epochs):
            
            # Start timer
            a = time.time()
            
            # Define the closure function that consists of resetting the
            # gradient buffer, loss function calculation, and backpropagation
            # The closure function is necessary for LBFGS optimizer, because
            # it requires multiple function evaluations
            # The closure function returns the loss value
            def closure():
                
                # Set the model to train mode
                self.model.train()
                
                # Reset the gradient buffer
                self.optimizer.zero_grad()
                
                # Loss function calculation
                lstm_train_loss = self.evaluate_train(inputs_swl_train,
                                                      inputs_train, target_train)
                
                # Backpropagate the loss function calculation
                lstm_train_loss.backward()
                
                return lstm_train_loss
            
            # Update the model parameters
            self.optimizer.step(closure)
            
            # Validate the model
            self.validate(inputs_swl_vali, inputs_vali, target_vali)
            
            # Write the loss values to the tensorboard log file
            self.tb.add_scalar('training_loss', self.train_loss[-1], epoch)
            self.tb.add_scalar('validation_loss', self.vali_loss[-1], epoch)
            
            # Stop the timer
            b = time.time()
            
            # Print out the epoch status
            print('Training: Epoch [%d/%d], Training Loss: %.4f, Validation Loss: %.4f, Runtime: %.4f secs'
                  %(epoch + 1, self.cfg.epochs, self.train_loss[-1], self.vali_loss[-1], b - a))
            
            
            # If the validation loss is lower than the best loss value,
            # update the best loss and save the model
            if self.vali_loss[-1] < self.best_vali:
                self.best_vali = self.vali_loss[-1]
                
                if self.cfg.save_model:
                    thread = Thread(target=self.save_model_to_file(
                        epoch))
                    thread.start()
                    
    
    
    def minibatch_training(self, train_loader, vali_loader):
        """
        This function trains the model with the mini batch method
        """
        
        # Set the number of threads for this program to one
        torch.set_num_threads(1)
        
        # Obtain the whole validation dataset
        inputs_swl_vali, inputs_vali, target_vali = next(iter(vali_loader))
        
        for epoch in range (self.start_epoch, self.cfg.epochs):
            
            # Start timer
            a = time.time()
            
            # Obtain mini batches of samples from the training data loader
            for batch_idx, (inputs_swl_train, inputs_train, target_train) in enumerate(train_loader):
                
                # Set the model to train mode
                self.model.train()
                
                # Reset the gradient buffer
                self.optimizer.zero_grad()
                
                # Loss function calculation
                lstm_train_loss = self.evaluate_train(inputs_swl_train,
                                                      inputs_train, target_train)
                
                # Backpropagate the loss function calculation
                lstm_train_loss.backward()
                
                # Update the model parameters
                self.optimizer.step()
                
            
            # Validate the model
            self.validate(inputs_swl_vali, inputs_vali, target_vali)
            
            # Write the loss values to the tensorboard log file
            self.tb.add_scalar('training_loss', self.train_loss[-1], epoch)
            self.tb.add_scalar('validation_loss', self.vali_loss[-1], epoch)
            
            # Stop the timer
            b = time.time()
            
            # Print out the epoch status
            print('Training: Epoch [%d/%d], Training Loss: %.4f, Validation Loss: %.4f, Runtime: %.4f secs'
                  %(epoch + 1, self.cfg.epochs, self.train_loss[-1], self.vali_loss[-1], b - a))
            
            
            # If the validation loss is lower than the best loss value,
            # update the best loss and save the model
            if self.vali_loss[-1] < self.best_vali:
                self.best_vali = self.vali_loss[-1]
                
                if self.cfg.save_model:
                    thread = Thread(target=self.save_model_to_file(
                        epoch))
                    thread.start()
                
    
    
    def evaluate_train(self, inputs_swl_train, inputs_train, target_train):
        """
        This function evaluates the loss function based on the current model
        """
        
        # Send the inputs and target to the corresponding device
        inputs_swl_train = inputs_swl_train.to(self.cfg.device)
        inputs_train = inputs_train.to(self.cfg.device)
        target_train = target_train.to(self.cfg.device)
        
        # Calculate the prediction using the training dataset
        lstm_output_train = self.model(inputs_swl_train, inputs_train)
        # Calculate the training loss
        lstm_train_loss = self.criterion(lstm_output_train, target_train)
        
        # Calculate the physics constraint when chosen to do so
        if self.cfg.physics_constraint:
        
            # Sample random indices for the physics constraint calculation
            # If full batch is used, memory requirement and computation time
            # would be too high
            rand_idx = torch.randint(0, self.cfg.batch_size_train,
                                     (self.cfg.num_samples_phys,1)).squeeze()
            
            
            # Precipitation
            # Modify the precipitation values to 0.50, 0.75, 1.00, 1.25, and 1.50
            # times the original values
            inp_temp_p = []
            lstm_output_train_p = []
        
            for i in range(5):
                inp_temp_p.append(inputs_train[rand_idx].clone().detach())
                
                # Modify the precipitation values
                inp_temp_p[i][:,:,0:2] = inputs_train[rand_idx,:,0:2].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                lstm_output_train_p.append(self.model(inputs_swl_train[rand_idx],
                                                inp_temp_p[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the precipitation values, the SWL prediction should
                # be higher
                if i >= 1:
                    lstm_train_loss += self.cfg.phys_mult * torch.mean(torch.relu(
                        lstm_output_train_p[i-1] - lstm_output_train_p[i]))
            
            
            # Evapotranspiration
            # Modify the evapotranspiration values to 0.50, 0.75, 1.00, 1.25, and 1.50
            # times the original values
            inp_temp_et = []
            lstm_output_train_et = []
            
            for i in range(5):
                inp_temp_et.append(inputs_train[rand_idx].clone().detach())
                
                # Modify the evapotranspiration values
                inp_temp_et[i][:,:,2:4] = inputs_train[rand_idx,:,2:4].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                lstm_output_train_et.append(self.model(inputs_swl_train[rand_idx],
                                                       inp_temp_et[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the evapotranspiration values, the SWL prediction
                # should be lower
                if i >= 1:
                    lstm_train_loss += self.cfg.phys_mult *  torch.mean(torch.relu(
                        lstm_output_train_et[i] - lstm_output_train_et[i-1]))
            
            
            # Well 5
            # Modify the well 5 pumping rate values to 0.50, 0.75, 1.00, 1.25,
            # and 1.50 times the original values
            inp_temp_w5 = []
            lstm_output_train_w5 = []
            
            temp_w5 = inputs_train[:,:,4:6]
            eps_w5 = 0.5
            pump_idx_w5 = torch.where(temp_w5 > eps_w5)
            if len(pump_idx_w5[0]) > self.cfg.num_samples_phys:
                rand_pump_idx_w5 = pump_idx_w5[0][torch.randint(0,len(pump_idx_w5[0]),(self.cfg.num_samples_phys,))]
            else:
                rand_pump_idx_w5 = pump_idx_w5[0]

            for i in range(5):
                inp_temp_w5.append(inputs_train[rand_pump_idx_w5].clone().detach())
                
                # Modify the well 5 pumping rate values
                inp_temp_w5[i][:,:,4:6] = inputs_train[rand_pump_idx_w5,:,4:6].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                lstm_output_train_w5.append(self.model(inputs_swl_train[rand_pump_idx_w5],
                                                       inp_temp_w5[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the well 5 pumping rate values, the SWL prediction
                # should be lower
                if i >= 1:
                    lstm_train_loss += self.cfg.phys_mult * torch.mean(torch.relu(
                        lstm_output_train_w5[i] - lstm_output_train_w5[i-1]))
                    
                    
            # Well 6
            # Modify the well 6 pumping rate values to 0.50, 0.75, 1.00, 1.25,
            # and 1.50 times the original values
            inp_temp_w6 = []
            lstm_output_train_w6 = []
            
            temp_w6 = inputs_train[:,:,6:]
            eps_w6 = 0.5
            pump_idx_w6 = torch.where(temp_w6 > eps_w6)
            if len(pump_idx_w6[0]) > self.cfg.num_samples_phys:
                rand_pump_idx_w6 = pump_idx_w6[0][torch.randint(0,len(pump_idx_w6[0]),(self.cfg.num_samples_phys,))]
            else:
                rand_pump_idx_w6 = pump_idx_w6[0]

            for i in range(5):
                inp_temp_w6.append(inputs_train[rand_pump_idx_w6].clone().detach())
                
                # Modify the well 6 pumping rate values
                inp_temp_w6[i][:,:,6:] = inputs_train[rand_pump_idx_w6,:,6:].clone().detach() * (0.5 + i*0.25)
                
                # Calculate the LSTM outputs with the modified values and store
                # them in a list
                lstm_output_train_w6.append(self.model(inputs_swl_train[rand_pump_idx_w6],
                                                       inp_temp_w6[i].to(self.cfg.device)).squeeze())
                
                # Add the constraint to the loss function
                # The higher the well 6 pumping rate values, the SWL prediction
                # should be lower
                if i >= 1:
                    lstm_train_loss += self.cfg.phys_mult * torch.mean(torch.relu(
                        lstm_output_train_w6[i] - lstm_output_train_w6[i-1]))
        
        # Append the loss function in the train_loss list and return the loss value
        self.train_loss.append(lstm_train_loss.item())
        
        return lstm_train_loss
                    
    
    
    def validate(self, inputs_swl_vali, inputs_vali, target_vali):
        """
        This function calculates the validation of the model
        """
        
        # For validation, the torch.no_grad() wrapper is needed so that the
        # forward propagation is not tracked, because backpropagation is not
        # needed
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Send the inputs and target to the corresponding device
            inputs_swl_vali = inputs_swl_vali.to(self.cfg.device)
            inputs_vali = inputs_vali.to(self.cfg.device)
            target_vali = target_vali.to(self.cfg.device)
            
            # Calculate the prediction using the validation dataset
            lstm_output_vali = self.model(inputs_swl_vali, inputs_vali)
    
            # Calculate the validation loss
            lstm_vali_loss = self.criterion(lstm_output_vali, target_vali)
            self.vali_loss.append(lstm_vali_loss.item())
            

    
    def save_model_to_file(self, epoch):
        """
        This function writes the model weights along with the network configuration
        and current performance to file
        """
        
        # Save model weights, optimizer state_dict, and epoch status to file
        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'loss_train': self.train_loss,
                 'loss_vali': self.vali_loss}
        torch.save(state, self.model_save_file)
        
        # Write the training performance and the configuration of the model to 
        # a file
        with open('m00_configuration.py', 'r') as f:
            cfg_file = f.read()
        
        
        output_string = cfg_file + "\n#\n# Training Performance\n\n"
    
        output_string += "CURRENT_EPOCH = " + str(epoch+1) + "\n"
        output_string += "EPOCHS = " + str(self.cfg.epochs) + "\n"
        output_string += "CURRENT_TRAINING_ERROR = " + \
                         str(self.train_loss[-1]) + "\n"
        output_string += "LOWEST_TRAINING_ERROR = " + \
                         str(min(self.train_loss)) + "\n"
        output_string += "CURRENT_VALIDATION_ERROR = " + \
                         str(self.vali_loss[-1]) + "\n"
        output_string += "LOWEST_VALIDATION_ERROR = " + \
                         str(min(self.vali_loss))
    
        # Save the configuration and current performance to file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
            _text_file.write(output_string)
