"""
M01_MAIN.PY
This is the main script to train and evaluate the LSTM model for the SWL
prediction
"""

from torch.utils.data import DataLoader
import m02_initialization as init
import m03_pre_process as preprocess
import m04_model as model
import m05_training as training
import m06_testing as testing
import m07_post_process as postprocess
import time


# Initialize configurations and hyperparameters
cfg = init.Initialize()


# Preprocess data
print('Preprocessing Data...')
print()

a = time.time()

dataset = preprocess.Initialize_Data(cfg)

train_data = preprocess.Load_Data(dataset,cfg,training=True)
vali_data = preprocess.Load_Data(dataset,cfg,validation=True)
test_data = preprocess.Load_Data(dataset,cfg,testing=True)

b = time.time()

train_loader = DataLoader(train_data, batch_size = cfg.batch_size_train, shuffle=False, drop_last=True)
vali_loader = DataLoader(vali_data, batch_size = cfg.batch_size_vali, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size = cfg.batch_size_test, shuffle=False, drop_last=True)

print('Preprocessing runtime: %.4f secs' %(b - a))
print()


# Initialize the LSTM model
lstm = model.ANN(cfg)

print(lstm)
print()


# Train the LSTM model

print('Start training...')
print()
trainer = training.Training(cfg, lstm)
trainer.train(train_loader, vali_loader)
print('Training finished...')
print()


# Evaluate (test) the trained model

print('Evaluating model...')
print()
tester = testing.Testing(cfg, trainer)
tester.evaluate(test_loader)


# Postprocess the data and plot figures

print('Postprocessing and plotting...')
print()

pp = postprocess.Post_process(cfg, trainer, tester, dataset, train_data, vali_data, test_data)

print('Finished!')
print()