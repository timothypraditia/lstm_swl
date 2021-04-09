## M00_CONFIGURATION.PY
## This script contains (hyper-)parameters required for the simulation and
## the network configuration


# DATA CONFIGURATION
data_filename = "df_not_norm.csv"
normalize = True
percentage_train = .64
percentage_vali = .18
percentage_test = .18
#Input size and prediction horizon:
in_swl = 1
in_p = 1
in_t = 1
in_sd = 1
in_rh = 1
in_wv = 1
in_p_forecast = 1 #houers
in_t_forecast = 1 #houers
in_sd_forecast = 1 #houers
in_rh_forecast = 1 #houers
in_wv_forecast = 1 #houers
forecast_horizon = 24 * 7 #hours
input_size = in_p + in_p_forecast + in_t + in_t_forecast + \
    in_sd + in_sd_forecast + in_rh + in_rh_forecast + in_wv + in_wv_forecast
print(input_size)
output_size = forecast_horizon


# MODEL NAME & SETTING
model_name = "m01_adam_01"
save_model = True
continue_training = False # Set to True to continue training with a saved model
device_name = "cpu" # Choose between "cpu" or "cuda"


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 2
hidden_size = 6
learning_rate = 0.01
phys_mult = 0.1 # Multiplier for the physical constraint
weight_decay = 1e-7
epochs = 1000
lbfgs_optim = False # Use L-BFGS as optimizer, else use ADAM
minibatch = False # Does not work with L-BFGS, will be overwritten

#For physical constraint:
physics_constraint = False
num_samples_phys = 5000
