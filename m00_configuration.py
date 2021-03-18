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
in_et = 1
in_w5 = 1
in_w6 = 1
in_p_forecast = 1 #hours
in_et_forecast = 1 #hours
in_w_forecast = 1 #hours
forecast_horizon = 24 * 7 #hours
input_size_swl = in_swl + in_p + in_et + in_p_forecast + in_et_forecast + \
    in_w5 + in_w_forecast + in_w6 + in_w_forecast
input_size = in_p + in_et + in_p_forecast + in_et_forecast + in_w5 + \
    in_w_forecast + in_w6 + in_w_forecast
output_size = forecast_horizon


# MODEL NAME & SETTING
model_name = "m01_adam_01"
save_model = True
continue_training = False # Set to True to continue training with a saved model
device_name = "cuda" # Choose between "cpu" or "cuda"


# NETWORK HYPER-PARAMETERS
num_lstm_layers = 2
hidden_size = 6
learning_rate = 0.01
phys_mult = 0.1 # Multiplier for the physical constraint
weight_decay = 1e-7
epochs = 500
lbfgs_optim = False # Use L-BFGS as optimizer, else use ADAM
minibatch = False # Does not work with L-BFGS, will be overwritten

#For physical constraint:
physics_constraint = True
num_samples_phys = 5000
