#!/usr/bin/env python
import os
import torch, sys, pyautogui
import numpy as np
import matplotlib.pyplot as plt

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from emg_regression.utils.torch_helper import TorchHelper
from emg_regression.approximators.lstm import LSTM
from emg_regression.utils.tools import *
from cfg.tools import get_subj_day_folder

# Flag for saving interface parameters and figures
SAVE = True

# Load params (subj, model_name)
params = load_yaml("cfg/model_params.yaml")

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
if params['subj'] == 'pendulum':
    data_path = f"{bomi_ws_path}data/pendulum/"
    model_path = f"{data_path}{params['model']['name']}"
else:
    subj_day_folder = get_subj_day_folder(params['subj'],params['date'])
    save_path = f"{subj_day_folder}training/"
    model_path = f"{subj_day_folder}model/{params['model']['name']}"
    _, _, train_idx, test_idx = load_train_test(save_path)
    data_path = f"{subj_day_folder}training/{params['modality']}/imuemg_{params['file']}.npy"
    data_loaded = np.load(data_path)
    fs = int(1/params['time_step'])
    # data_loaded = data_loaded[fs:,:,:]
    # stay with only 5 seconds
    data_loaded = data_loaded[fs:-fs,:,:]
    t    = data_loaded[:,0,0] - data_loaded[0,0,0]
    data = data_loaded[:,:,1:]
    data, input_dim, output_dim = select_IO(params, data)

train_data = data[:,train_idx,:]
test_data  = data[:,test_idx,:]
np.save(f"{subj_day_folder}testing/time", t)

NTrain, NTest = train_data.shape[1], test_data.shape[1]
print(f"NTrain: {NTrain}, NTest: {NTest}")

# Load params used in training the model
test_params = load_yaml(f"{subj_day_folder}model/{params['model']['name']}_config.yaml")
pad_len = test_params['pad_len']*test_params['window_size']
pad_len = 0

# Get input and output data for model
input_dim, output_dim = test_params['input_dim'], test_params['output_dim']
print('input_dim:', input_dim, ', output_dim:', output_dim)

# Normalize the data
# Normalize data for model training and save
# check_data(t,train_data[pad_len:,:,:],output_dim,N=8,lim=3)

# Save reference EMG and IMU NOT normalized
ref_emg = test_data[:,:,output_dim:].transpose(1,0,2)[:,pad_len:,:]
ref_imu = test_data[:,:,:output_dim].transpose(1,0,2)[:,pad_len:,:]

# emg mvc
emg_mvc = train_data[:,:,output_dim:].max((0,1))
emg_mvc_path  = f'{subj_day_folder}/interface/emg_mvc_train'

train_data = normalize_data(train_data, output_dim, 
                      test_params['normalize_input'], 
                      test_params['normalize_output'],
                      save_path, save=False, load=True)

test_data = normalize_data(test_data, output_dim, 
                      test_params['normalize_input'], 
                      test_params['normalize_output'],
                      save_path, save=False, load=True)

# Check data
# check_data(t,train_data,output_dim,N=8,lim=3)
# pad_len=5
check_data(t,test_data[:,:,:],output_dim,N=NTest,lim=5) # check reference signals
# save them too

# Save reference EMG and IMU normalized
ref_emg_norm = test_data[:,:,output_dim:].transpose(1,0,2)[:,pad_len:,:]
ref_imu_norm = test_data[:,:,:output_dim].transpose(1,0,2)[:,pad_len:,:]

# Remove for testing the first second of data
t2 = t
# t2 = t[int(1/params['time_step']):]
# train_data = train_data[int(1/params['time_step']):,:,:]
# test_data  = test_data[int(1/params['time_step']):,:,:]

# Set input and output for specific LSTM
input_train  = train_data if test_params['autoregressive'] else train_data[:,:,output_dim:]
output_train = train_data[:,:,:output_dim]
input_test  = test_data if test_params['autoregressive'] else test_data[:,:,output_dim:]
output_test = test_data[:,:,:output_dim]

# Window data for LSTM
delay_samples = 1 if test_params['autoregressive'] else 0
train_x, train_y = preprocess_data_for_lstm(
                         input_train, output_train, 
                         test_params['window_size'], test_params['window_step'], 
                         delay_samples, device)
test_x, test_y = preprocess_data_for_lstm(
                         input_test, output_test, 
                         test_params['window_size'], test_params['window_step'], 
                         delay_samples, device)

# Load model
model = LSTM(input_size=input_dim, 
                hidden_dim=test_params['hidden_dim'], 
                pre_output_size=test_params['preoutput_size'],
                output_size=output_dim, 
                dropout=test_params['dropout'],
                n_layers=test_params['num_layers']).to(device)

TorchHelper.load(model, model_path, device)
print('Model loaded:',model_path)
model.eval()

# Training set prediction
window_size = params['window_size']
ypred_train_ol = predict_motion(train_data,model,output_dim,window_size,device,autoregress=False)
ypred_train_cl = predict_motion(train_data,model,output_dim,window_size,device,autoregress=True)

# MSE train
ylabels = [r'$\theta$ (rad)', r'$\phi$ (rad)', r'$\dot{\theta}$ (rad/s)', r'$\dot{\phi}$ (rad/s)']
ytrain = train_data[:,:,:output_dim].transpose(1,0,2)
mse_ol = get_mse(ytrain,ypred_train_ol)
mse_cl = get_mse(ytrain,ypred_train_cl)
ave_mse_ol, ave_mse_cl = mse_ol.mean(), mse_cl.mean()

# Testing set
ypred_ol_test = predict_motion(test_data,model,output_dim,window_size,device,autoregress=False)
ypred_cl_test = predict_motion(test_data,model,output_dim,window_size,device,autoregress=True)

# MSE test
ytest = test_data[:,:,:output_dim].transpose(1,0,2)
mse_test_ol = get_mse(ytest,ypred_ol_test)
mse_test_cl = get_mse(ytest,ypred_cl_test)
ave_mse_test_ol, ave_mse_test_cl = mse_test_ol.mean(), mse_test_cl.mean()

#colors
c_train = sns.color_palette('hls', 500)[::10]
c_test = sns.color_palette('hls', 100)[::20]

fig, ax = plt.subplots(output_dim,2,figsize=(8,8))
for j in range(output_dim):
    for traj_id in range(NTrain):
        ax[j,0].plot(t2,mse_ol[traj_id,:,j],'-o',color=c_train[traj_id],markersize=1,alpha=0.5,label=f'Traj {traj_id+1}')
    for traj_id in range(NTest):
        ax[j,1].plot(t2,mse_test_ol[traj_id,:,j],'-o',color=c_test[traj_id],markersize=1,alpha=0.5,label=f'Traj {traj_id+1}')
    ax[j,0].plot(t2,mse_ol.mean(0)[:,j],'-o',color='k',markersize=0.5)
    ax[j,1].plot(t2,mse_test_ol.mean(0)[:,j],'-o',color='k',markersize=0.5)
    ax[j,0].set_ylabel(ylabels[j])
    ax[j,0].grid(); ax[j,1].grid()
ax[0,0].set_title('MSE Training'); ax[-1,0].set_xlabel('Time (s)')
ax[0,1].set_title('MSE Testing'); ax[-1,1].set_xlabel('Time (s)')
plt.tight_layout()
if SAVE:
    plt.savefig(f"{subj_day_folder}figs/mse_traj_{params['model']['name']}.png",format='png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()


# plot trajectories (normalized angles)
fig, axes = plt.subplots(2,2,figsize=(10,8))
plot_2Dtraj(axes[0,0], ytrain, ypred_train_ol, title=f'Open-loop Training prediction (N={NTrain}, MSE={ave_mse_ol.round(5)})',   colors=c_train)
plot_2Dtraj(axes[0,1], ytrain, ypred_train_cl,  title=f'Closed-loop Training prediction (N={NTrain})',colors=c_train)
plot_2Dtraj(axes[1,0], ytest, ypred_ol_test, title=f'Open-loop Testing prediction (N={NTest}, MSE={ave_mse_test_ol.round(5)})',  colors=c_test)
plot_2Dtraj(axes[1,1], ytest, ypred_cl_test, title=f'Closed-loop Testing prediction (N={NTest})',colors=c_test)
plt.tight_layout()
if SAVE:
    plt.savefig(f"{subj_day_folder}figs/traj_pred_{params['model']['name']}.png",format='png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()

# average MSE
ylabels_nu = [r'$\theta$', r'$\phi$', r'$\dot{\theta}$', r'$\dot{\phi}$']
c_base_kin = sns.color_palette("hls", 4)
# c_test = sns.color_palette('hls', 100)[::10]

titles = ['Training','Testing']
mses = [mse_ol.mean(0), mse_test_ol.mean(0)]

fig, axes = plt.subplots(1,2,figsize=(12,4))
for i,ax in enumerate(axes.flatten()):
    for j in range(output_dim):
        ax.plot(t2, mses[i][:,j],'-o',markersize=1,alpha=0.6,label=ylabels_nu[j],linewidth=1.5, color=c_base_kin[j])
    ax.set_title(titles[i]); ax.grid(); 
axes[0].set_ylabel('MSE')
axes[0].set_xlabel('Time (s)'); axes[1].set_xlabel('Time (s)')
axes[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.8))
plt.tight_layout()
if SAVE:
    plt.savefig(f"{subj_day_folder}figs/mse_ave_{params['model']['name']}.png",format='png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()


# """ >> Map from body to screen and save data """

# First, get the predicted trunk angles in radians
mu_y  = np.load(f"{save_path}/mu_y.npy")
std_y = np.load(f"{save_path}/std_y.npy")
ytrue_test_deg = ((ytest * std_y) + mu_y)*180/np.pi
# ypred_test_deg = ((ypred_ol_test * std_y) + mu_y)*180/np.pi
ypred_test_deg = ((ypred_cl_test * std_y) + mu_y)*180/np.pi

# plot trajectories (normalized angles)
# fig, ax = plt.subplots(figsize=(10,6))
# plot_2Dtraj(ax, ytrue_test_deg, ypred_test_deg, title=f'Testing prediction (N={NTest})',colors=c_test)
# plt.tight_layout()
# plt.show()


""" Map predicted trajectories to screen (measured and predicted)"""
# Map 
H_imu_deg = np.load(f'{subj_day_folder}interface/H_imu_deg.npy')
d1, d2 = pyautogui.size() # must be run in the same device of gui
margin = 0
traj_true_screen = np.array([map_pred_to_screen(traj,d1,d2,H_imu_deg,home_loc='center',margin=margin) 
                             for traj in ytrue_test_deg])
traj_pred_screen = np.array([map_pred_to_screen(traj,d1,d2,H_imu_deg,home_loc='center',margin=margin) 
                             for traj in ypred_test_deg])

# Normalized EMG by MVC
# emg_mvc = np.load(f"{subj_day_folder}interface/emg_mvc.npy")

# Plot data for ref
plot_reference(y=traj_pred_screen, ytrue=traj_true_screen, 
               t=t-t[0], ref_emg=ref_emg, 
               xlim=[0,d1],ylim=[d2,0],ylim_emg=None)
if SAVE:
    plt.savefig(f"{subj_day_folder}figs/reference_test_MVCnorm.png",format='png', dpi=300, bbox_inches='tight', facecolor='w')
plt.show()


# # Things to save
if SAVE:
    np.save(f"{subj_day_folder}testing/ref_emg", ref_emg)
    np.save(f"{subj_day_folder}testing/ref_imu", ref_imu)
    np.save(f"{subj_day_folder}testing/ref_emg_norm", ref_emg_norm)
    np.save(f"{subj_day_folder}testing/ref_imu_norm", ref_imu_norm)
    # np.save(emg_mvc_path,emg_mvc)
    print("Saved: ref_emg, ref_imu")
    print("Saved: ref_emg_norm, ref_imu_norm")

    # Interface
    np.save(f'{subj_day_folder}interface/screen',(d1,d2))
    np.save(f'{subj_day_folder}interface/traj_screen_true',traj_true_screen)
    np.save(f'{subj_day_folder}interface/traj_screen_pred',traj_pred_screen)
    np.save(f'{subj_day_folder}interface/ytrue_test_deg',ytrue_test_deg)
    np.save(f'{subj_day_folder}interface/ypred_test_deg',ypred_test_deg)
    print("Saved: screen_dim, traj_screen true and pred")

# # select trajectories
# # idx_select = np.array([1,3,4,5,6])-1
# ypred_test1 = ypred_test[idx_select,:,:]
