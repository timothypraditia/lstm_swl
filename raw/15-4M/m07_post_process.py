"""
M07_POST_PROCESS.PY
This script contains the post processing class that is used to post process and
plot the results
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


class Post_process:
    
    def __init__(self, cfg, trainer, tester, dataset, train_data, vali_data, test_data):
        """
        Constructor
        """
        
        self.cfg = cfg
        self.trainer = trainer
        self.tester = tester
        self.train_data = train_data
        self.vali_data = vali_data
        self.test_data = test_data
        
        # Calculate the prediction using the training, validation, and testing
        # dataset
        self.prediction_train = self.trainer.model(self.train_data.inputs_swl.to(self.cfg.device),
                                         self.train_data.inputs.to(self.cfg.device))
        self.prediction_vali = self.trainer.model(self.vali_data.inputs_swl.to(self.cfg.device),
                                        self.vali_data.inputs.to(self.cfg.device))
        self.prediction_test = self.trainer.model(self.test_data.inputs_swl.to(self.cfg.device),
                                        self.test_data.inputs.to(self.cfg.device))
        
        # Denormalize the data if necessary
        if self.cfg.normalize:
            self.train_data.target = self.denormalize_data(self.train_data.target)
            self.vali_data.target = self.denormalize_data(self.vali_data.target)
            self.test_data.target = self.denormalize_data(self.test_data.target)
            self.prediction_train = self.denormalize_data(self.prediction_train)
            self.prediction_vali = self.denormalize_data(self.prediction_vali)
            self.prediction_test = self.denormalize_data(self.prediction_test)
            dataset.tar_swl = self.denormalize_data(dataset.tar_swl)
        
        # Plot training and validation loss vs epochs
        self.plot_loss()
        
        # Define the horizon_dict that corresponds to 1, 3, 5, and 7 days horizon
        self.horizon_dict = {0:1, 1:3, 2:5, 3:7}
        
        # Initialize tensor arrays to store prediction statistics
        self.max_error_train = torch.zeros(len(self.horizon_dict))
        self.max_error_vali = torch.zeros(len(self.horizon_dict))
        self.max_error_test = torch.zeros(len(self.horizon_dict))
        self.mse_train = torch.zeros(len(self.horizon_dict))
        self.mse_vali = torch.zeros(len(self.horizon_dict))
        self.mse_test = torch.zeros(len(self.horizon_dict))
        
        for i in range(4):
            # Get the prediction statistics
            self.compute_stats(i)
            
            # Plot prediction for different forecast horizon: 1, 3, 5, and 7 days
            self.plot_prediction(self.horizon_dict[i]*24, dataset)
        
        # Plot the prediction comparison using different input values
        self.plot_phys()
        
        # Update the configuration and performance file with the prediction
        # statistics
        self.update_config_log()
        
        # Save the predictions into .csv files
        self.save_data()
        
            
            
    def denormalize_data(self, data):
        """
        This function denormalize the data into their true absolute values 
        using the maximum and minimum value
        """
        
        data = data * (self.cfg.max_swl - self.cfg.min_swl) + self.cfg.min_swl
        
        return data
    
    

    def plot_loss(self):
        """
        This function plots the loss vs epochs for the training and validation
        """
        
        fig, (p1, p2) = plt.subplots(2, 1, sharey=False, figsize=(16, 8))
        
        # Plot the training loss
        plt.subplots_adjust(hspace=0.5)
        p1.plot(np.array(self.trainer.train_loss))
        p1.set_title('Training Loss vs Epoch', size=18)
        p1.tick_params(axis='x', labelsize=16)
        p1.tick_params(axis='y', labelsize=16)
        p1.set_xlabel('Epoch', size=18)
        p1.set_ylabel('Training Loss', size=18)
        
        # Plot the validation loss
        p2.plot(np.array(self.trainer.vali_loss))
        p2.set_title('Validation Loss vs Epoch', size=18)
        p2.tick_params(axis='x', labelsize=16)
        p2.tick_params(axis='y', labelsize=16)
        p2.set_xlabel('Epoch', size=18)
        p2.set_ylabel('Validation Loss', size=18)
        
        # Save the figure
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name + '_loss_plot.png')
            plt.close()
    

    def plot_prediction(self, plot_horizon, dataset):
        """
        This function plots the prediction of the LSTM model compared with the
        target values separately for training, validation, and testing
        """
        
        # Extract the start and end of the timestamps from the data converted
        # into the np.datetime format
        start_hour_full = np.datetime64(dataset.data.iloc[0,0])
        end_hour_full = np.datetime64(dataset.data.iloc[len(dataset.tar_swl)-1,0])
        self.timestamps_full = np.arange(start_hour_full, end_hour_full, dtype='datetime64[h]')
        
        
        # Slice the timestamp for the full target plot, starting from t = t0 + plot_horizon
        # because the first available prediction is at this time step
        timestamp_target = self.timestamps_full[plot_horizon:
                            len(dataset.tar_swl) - (self.cfg.forecast_horizon - plot_horizon) - 1]
    
        # Slice the timestamp for the training, validation, and testing prediction
        # plot, starting from time step t = t_start + plot_horizon for each
        # stage, up to time step t = t_end - (forecast_horizon - plot_horizon) - 1
        timestamp_train = self.timestamps_full[self.train_data.count_start + plot_horizon:
                            self.train_data.count_end - (self.cfg.forecast_horizon - plot_horizon) - 1]
        
        timestamp_vali = self.timestamps_full[self.vali_data.count_start + plot_horizon:
                            self.vali_data.count_end - (self.cfg.forecast_horizon - plot_horizon) - 1]
            
        timestamp_test = self.timestamps_full[self.test_data.count_start + plot_horizon:
                            self.test_data.count_end - (self.cfg.forecast_horizon - plot_horizon) - 1]
        
        # Slice the target values corresponding to the timestamps
        target = dataset.tar_swl[plot_horizon:
                        len(dataset.tar_swl) - (self.cfg.forecast_horizon - plot_horizon) - 1, 0]
        
        # Slice the target values for each stage separately
        target_train = self.train_data.target[:,plot_horizon-1,:]
        target_vali = self.vali_data.target[:,plot_horizon-1,:]
        target_test = self.test_data.target[:,plot_horizon-1,:]
    
        # Prepare the prediction for training, validation, and testing
        # Has to be sent to cpu for plotting purposes
        prediction_train = self.prediction_train[:,plot_horizon-1,:].cpu().detach()
        prediction_vali = self.prediction_vali[:,plot_horizon-1,:].cpu().detach()
        prediction_test = self.prediction_test[:,plot_horizon-1,:].cpu().detach()
            
            
        # timestamp_plot = timestamps_full[data.count_start + plot_horizon:
        #                                  data.count_end - (cfg.forecast_horizon - plot_horizon) - 1]
            
        # Plot in a combined plot for all stages
        fig, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_target, target, color = color, label='Measurement')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        color = 'tab:green'
        ax1.plot(timestamp_train, prediction_train, color=color, label='Train Prediction')
        color = 'tab:orange'
        ax1.plot(timestamp_vali, prediction_vali, color=color, label='Validation Prediction')
        color = 'tab:red'
        ax1.plot(timestamp_test, prediction_test, color=color, label='Test Prediction')
        plt.title('SWL Measurement and Prediction', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        '_combined_' + str(plot_horizon//24) + 'days_plot.png')
            plt.close()
            
        
        # Plot the training prediction separately
        fig, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_train, target_train, color = color, label='Measurement')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        color = 'tab:red'
        ax1.plot(timestamp_train, prediction_train, color=color, label='Train Prediction')
        plt.title('SWL Measurement and Prediction', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        '_train_' + str(plot_horizon//24) + 'days_plot.png')
            plt.close()
        
        # Plot the validation prediction separately
        fig, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_vali, target_vali, color = color, label='Measurement')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        color = 'tab:red'
        ax1.plot(timestamp_vali, prediction_vali, color=color, label='Validation Prediction')
        plt.title('SWL Measurement and Prediction', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        '_vali_' + str(plot_horizon//24) + 'days_plot.png')
            plt.close()
        
        # Plot the training prediction separately
        fig, ax1 = plt.subplots(figsize=(16, 8))
        color = 'tab:blue'
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_test, target_test, color = color, label='Measurement')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        color = 'tab:red'
        ax1.plot(timestamp_test, prediction_test, color=color, label='Testing Prediction')
        plt.title('SWL Measurement and Prediction', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        '_test_' + str(plot_horizon//24) + 'days_plot.png')
            plt.close()
            
    
    def plot_phys(self):
        """
        This function plots the comparison of the LSTM predictions when the
        input values are modified, to check physical consistency and plausibility
        """
        
        # Extract the random index used during testing for the physical error
        # calculation to find the corresponding timestamps
        start_idx = self.tester.rand_idx[0]
        
        # Slice the timestamp for the plotting, corresponding to the random
        # index used during the evaluation in testing
        timestamp_plot = self.timestamps_full[self.test_data.count_start + start_idx:
                            self.test_data.count_start + start_idx + self.cfg.forecast_horizon]
        
        # Extract the LSTM prediction for different precipitation values
        prediction_p_low = self.tester.lstm_output_test_p[0][0].cpu().detach()
        prediction_p_mid = self.tester.lstm_output_test_p[2][0].cpu().detach()
        prediction_p_high = self.tester.lstm_output_test_p[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different air temperature values
        prediction_t_low = self.tester.lstm_output_test_t[0][0].cpu().detach()
        prediction_t_mid = self.tester.lstm_output_test_t[2][0].cpu().detach()
        prediction_t_high = self.tester.lstm_output_test_t[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different sunshine duration values
        prediction_sd_low = self.tester.lstm_output_test_sd[0][0].cpu().detach()
        prediction_sd_mid = self.tester.lstm_output_test_sd[2][0].cpu().detach()
        prediction_sd_high = self.tester.lstm_output_test_sd[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different relative humidity values
        prediction_rh_low = self.tester.lstm_output_test_rh[0][0].cpu().detach()
        prediction_rh_mid = self.tester.lstm_output_test_rh[2][0].cpu().detach()
        prediction_rh_high = self.tester.lstm_output_test_rh[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different wind velocity values
        prediction_wv_low = self.tester.lstm_output_test_wv[0][0].cpu().detach()
        prediction_wv_mid = self.tester.lstm_output_test_wv[2][0].cpu().detach()
        prediction_wv_high = self.tester.lstm_output_test_wv[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different well 5 pumping rate values
        prediction_w5_low = self.tester.lstm_output_test_w5[0][0].cpu().detach()
        prediction_w5_mid = self.tester.lstm_output_test_w5[2][0].cpu().detach()
        prediction_w5_high = self.tester.lstm_output_test_w5[4][0].cpu().detach()
        
        # Extract the LSTM prediction for different well 6 pumping rate values
        prediction_w6_low = self.tester.lstm_output_test_w6[0][0].cpu().detach()
        prediction_w6_mid = self.tester.lstm_output_test_w6[2][0].cpu().detach()
        prediction_w6_high = self.tester.lstm_output_test_w6[4][0].cpu().detach()
        
        
        # Plot the comparison of different precipitation values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_p_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_p_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_p_high, label='High')
        plt.title('SWL Prediction for Different Precipitation', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_p.png')
            plt.close()
        
        # Plot the comparison of different air temperature values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_t_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_t_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_t_high, label='High')
        plt.title('SWL Prediction for Different Air Temperatures', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_t.png')
            plt.close()
        
        # Plot the comparison of different sunshine duration values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_sd_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_sd_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_sd_high, label='High')
        plt.title('SWL Prediction for Different Sunshine Duration', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_sd.png')
            plt.close()
        
        # Plot the comparison of different relative humidity values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_rh_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_rh_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_rh_high, label='High')
        plt.title('SWL Prediction for Different Relative Humidity', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_rh.png')
            plt.close()
        
        # Plot the comparison of different wind velocity values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_wv_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_wv_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_wv_high, label='High')
        plt.title('SWL Prediction for Different Wind Velocity', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_wv.png')
            plt.close()
            
        # Plot the comparison of different well 5 pumping rate values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_w5_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_w5_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_w5_high, label='High')
        plt.title('SWL Prediction for Different Well 5 Pumping Rate', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_w5.png')
            plt.close()
        
        # Plot the comparison of different well 6 pumping rate values
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.set_xlabel('Timestamp', size=18)
        ax1.set_ylabel('Ground water level [m]', size=18)
        ax1.plot(timestamp_plot, prediction_w6_low, label='Low')
        ax1.tick_params(axis='y', labelcolor='k', labelsize=18)
        ax1.tick_params(axis='x', labelcolor='k', rotation=30, labelsize=18)
        ax1.plot(timestamp_plot, prediction_w6_mid, label='Medium')
        ax1.plot(timestamp_plot, prediction_w6_high, label='High')
        plt.title('SWL Prediction for Different Well 6 Pumping Rate', size=18)
        ax1.legend(loc='upper right', frameon=False, fontsize=18)
        fig.tight_layout()
        if self.cfg.save_model:
            fig.savefig(self.cfg.model_path + '\\' + self.cfg.model_name +
                        'compare_w6.png')
            plt.close()
    
    
    def compute_stats(self, horizon_idx):
        """
        This function computes the statistics of the prediction and update the
        configuration and performance file
        """
        
        # Calculate the absolute maximum error between the prediction and the
        # target for training, validation, and testing
        self.max_error_train[horizon_idx] = max(abs(
            self.prediction_train[:,self.horizon_dict[horizon_idx]*24-1,:].cpu().detach()
            - self.train_data.target[:,self.horizon_dict[horizon_idx]*24-1,:]))
        
        self.max_error_vali[horizon_idx] = max(abs(
            self.prediction_vali[:,self.horizon_dict[horizon_idx]*24-1,:].cpu().detach()
            - self.vali_data.target[:,self.horizon_dict[horizon_idx]*24-1,:]))
        
        self.max_error_test[horizon_idx] = max(abs(
            self.prediction_test[:,self.horizon_dict[horizon_idx]*24-1,:].cpu().detach()
            - self.test_data.target[:,self.horizon_dict[horizon_idx]*24-1,:]))
        
        # Calculate the unnormalized MSE between the prediction and the target
        # for training, validation, and testing
        self.mse_train[horizon_idx] = self.trainer.criterion(
            self.prediction_train[:,self.horizon_dict[horizon_idx]*24-1,:],
            self.train_data.target[:,self.horizon_dict[horizon_idx]*24-1,:].to(
                self.cfg.device)).item()
        
        self.mse_vali[horizon_idx] = self.trainer.criterion(
            self.prediction_vali[:,self.horizon_dict[horizon_idx]*24-1,:],
            self.vali_data.target[:,self.horizon_dict[horizon_idx]*24-1,:].to(
                self.cfg.device)).item()
        
        self.mse_test[horizon_idx] = self.trainer.criterion(
            self.prediction_test[:,self.horizon_dict[horizon_idx]*24-1,:],
            self.test_data.target[:,self.horizon_dict[horizon_idx]*24-1,:].to(
                self.cfg.device)).item()
        
        
    def update_config_log(self):
        """
        This function updates the configuration and performance log file
        """
        
        # Open the configuration and performance log file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'r') as f:
            cfg_file = f.read()
            
        output_string = cfg_file + "\n#\n# Unnormalized Prediction Statistics\n\n"
        
        for i in range(len(self.horizon_dict)):
            output_string += "TRAINING_MAXIMUM_ERROR_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.max_error_train[i].item()) + " M\n"
            output_string += "VALIDATION_MAXIMUM_ERROR_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.max_error_vali[i].item()) + " M\n"
            output_string += "TESTING_MAXIMUM_ERROR_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.max_error_test[i].item()) + " M\n"
            output_string += "TRAINING_MSE_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.mse_train[i].item()) + " M\n"
            output_string += "VALIDATION_MSE_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.mse_vali[i].item()) + " M\n"
            output_string += "TESTING_MSE_" + str(self.horizon_dict[i]) + \
                            "_DAY(S) = " + str(self.mse_test[i].item()) + " M\n\n"
    
        # Save the updated performance metrics into the file
        with open(self.cfg.model_path + '\\' + self.cfg.model_name + '_cfg_and_performance.txt', 'w') as _text_file:
            _text_file.write(output_string)
    
    
    def save_data(self):
        """
        This function saves the prediction values to .csv files
        """
        
        # Save the training prediction to a .csv file
        pd.DataFrame(np.squeeze(self.prediction_train.squeeze().cpu().detach().numpy())).to_csv(
            self.cfg.model_path + '\\train_prediction.csv', sep = "\t", float_format = '%.4f')
        
        # Save the validation prediction to a .csv file
        pd.DataFrame(np.squeeze(self.prediction_vali.squeeze().cpu().detach().numpy())).to_csv(
            self.cfg.model_path + '\\vali_prediction.csv', sep = "\t", float_format = '%.4f')
        
        # Save the testing prediction to a .csv file
        pd.DataFrame(np.squeeze(self.prediction_test.squeeze().cpu().detach().numpy())).to_csv(
            self.cfg.model_path + '\\test_prediction.csv', sep = "\t", float_format = '%.4f')
        