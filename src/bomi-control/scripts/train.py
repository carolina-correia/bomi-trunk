#!/usr/bin/env python
import os
import torch, sys, ast
import numpy as np
import matplotlib.pyplot as plt
# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from emg_regression.utils.torch_helper import TorchHelper
from torch.utils.data import TensorDataset, DataLoader
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.tools import *
from cfg.tools import get_subj_day_folder
import seaborn as sns

# set torch device
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load params
params = load_yaml("cfg/model_params.yaml")
print_params(params)

# Load data
if params['subj'] == 'pendulum':
    data_path = f"{bomi_ws_path}data/pendulum/"
    data = np.load(f"{data_path}pendulum.npy")
    model_path = f"{data_path}{params['model']['name']}"
    data, input_dim, output_dim = select_IO_pendulum(params, data)
    save_folder = data_path
else:
    subj_day_folder = get_subj_day_folder(params['subj'],params['date'])
    data_path = f"{subj_day_folder}training/{params['modality']}/imuemg_{params['file']}.npy"
    model_path = f"{subj_day_folder}model/{params['model']['name']}"
    print(data_path, model_path)
    data_loaded = np.load(data_path)    
    # stay with only 5 seconds
    fs = int(1/params['time_step'])
    data_loaded = data_loaded[fs:-fs,:,:]
    t    = data_loaded[:,0,0] - data_loaded[0,0,0]
    data = data_loaded[:,:,1:]
    data, input_dim, output_dim = select_IO(params, data)
    save_folder = f"{subj_day_folder}training/"

# Select input and output data for model
total_nb_trajectories = data.shape[1]
nb_ch = input_dim - output_dim

print('input_dim:', input_dim, ', output_dim:', output_dim)
print('data:', data.shape)
print('Total nb_trajectories:', total_nb_trajectories)

# Check data
check_data(t,data,output_dim,N=8)

# Pad input, if selected
t_pad, data, pad_len = add_padding(data,params['train']['padding'],params['window_size'],
                            t, params['time_step'],
                            traj_first=False)

# Split sample trajectories in training and testing and save them
if params['train']['split'] == 'random':
    train_data_raw, test_data_raw, train_idx, test_idx = select_train_test(data,train_perc=params['train']['percentage'],random=params['train']['random'],train_idx=params['train']['traj_idx'],test_idx=params['test']['traj_idx']) # train_idx=None,test_idx=None)
    print(f"NTrain: {train_data_raw.shape[1]}, NTest: {test_data_raw.shape[1]}")
    plot_train_test(train_data_raw,test_data_raw,lim=0.8)
    question = "Do you want to proceed with this split? (y/n): "
    proceed = input(question).lower()
    while proceed != 'y':
        train_data_raw, test_data_raw, train_idx, test_idx = select_train_test(data,train_perc=params['train']['percentage'],random=params['train']['random'],train_idx=params['train']['traj_idx'],test_idx=params['test']['traj_idx'])
        plot_train_test(train_data_raw, test_data_raw,lim=0.8)
        proceed = input(question).lower()
    # save before normalizing
    save_train_test(save_folder,train_data_raw, test_data_raw,train_idx,test_idx)
    
if params['train']['split'] == 'cmw':
    idx_sort = select_test_CMW(data, t, output_dim, nb_options=5)
    request = "Select the trajectory indexes for each channel (default: [0,0,0,0]): "
    user_input = input(request)
    idx_selected = [0,0,0,0] if not user_input else ast.literal_eval(user_input)

    test_idx = [idx_sort[idx_selected[ch],ch] for ch in range(nb_ch)]
    train_idx = list(set(list(range(total_nb_trajectories))) - set(test_idx))
    train_data_raw, test_data_raw = data[:,train_idx,:], data[:,test_idx,:]

    plot_train_test(train_data_raw, test_data_raw,lim=0.8)
    question = "Do you want to proceed with this split? (y/n): "
    proceed = input(question).lower()
    while proceed != 'y':
        request = "Select the trajectory indexes for each channel (default: [0,0,0,0]): "
        idx_selected = ast.literal_eval(input(request))
        test_idx = [idx_sort[idx_selected[ch],ch] for ch in range(nb_ch)]
        train_idx = list(set(list(range(total_nb_trajectories))) - set(test_idx))
        train_data_raw, test_data_raw = data[:,train_idx,:], data[:,test_idx,:]
        plot_train_test(train_data_raw, test_data_raw,lim=0.8)
        proceed = input(question).lower()
    # save before normalizing
    save_train_test(save_folder,train_data_raw, test_data_raw,train_idx,test_idx)

else:
    # Load previous split
    _, _, train_idx, test_idx = load_train_test(save_folder)
    train_data_raw, test_data_raw = data[:,train_idx,:], data[:,test_idx,:]

train_data, test_data = train_data_raw, test_data_raw
NTrain, NTest = train_data.shape[1], test_data.shape[1]
print(f"NTrain: {NTrain}, NTest: {NTest}")

# Normalization of trunk ROM
min_theta, max_theta = abs(train_data[:,:,0].min()), abs(train_data[:,:,0].max())
min_phi,   max_phi   = abs(train_data[:,:,1].min()), abs(train_data[:,:,1].max())
max_side = max(min_theta,max_theta)
H_rad = np.array([[max_side,max_side],
                  [min_phi,  max_phi]])
H_deg = H_rad*180/np.pi

# Save new calibration matrix based on training data
np.save(f"{save_folder}/time", t)
np.save(f'{subj_day_folder}interface/H_imu_deg',H_deg)
np.save(f'{subj_day_folder}interface/H_imu_rad',H_rad)

# Normalize data for model training and save
train_data = normalize_data(train_data, output_dim, 
                      params['train']['normalize_input'], 
                      params['train']['normalize_output'],
                      save_folder, save=True, load=False)
# Check data
# check_data(t_pad,train_data,output_dim,N=8,lim=3)

# Set input and output for specific LSTM
input_train  = train_data if params['model']['autoregressive'] else train_data[:,:,output_dim:]
output_train = train_data[:,:,:output_dim]

# Window data for LSTM
delay_samples = 1 if params['model']['autoregressive'] else 0
train_x, train_y = preprocess_data_for_lstm(
                         input_train, output_train, 
                         params['window_size'], params['window_step'], 
                         delay_samples, device)
# Model
model = LSTM(input_size=input_dim, 
                hidden_dim=params['model']['hidden_dim'], 
                pre_output_size=params['model']['preoutput_size'],
                output_size=output_dim, 
                dropout=params['model']['dropout'],
                n_layers=params['model']['num_layers']).to(device)

# Load previous model
if params['train']['load']:
    TorchHelper.load(model, model_path, device)
model.train()

# Batch loader
dataset     = TensorDataset(train_x, train_y)
batch_size  = params['train']['batch_size']
batch_size  = len(train_x) if batch_size == 0 else batch_size
data_loader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=params['train']['shuffle'])
    
# Train model
model, loss_log, epochs, training_time = train_model(model, params, data_loader)
plt.plot(loss_log)
plt.show()
print("Training time =", training_time)

# save model
TorchHelper.save(model, model_path)
print('Model saved to:',model_path)

# Save training parameters
params_dict = get_params_dict(params,input_dim,output_dim,delay_samples,NTrain)
save_to_yaml(f"{subj_day_folder}model/{params['model']['name']}_config.yaml", params_dict)
# save_to_yaml("cfg/test_config.yaml", params_dict)
