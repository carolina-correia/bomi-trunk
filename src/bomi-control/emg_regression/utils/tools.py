#!/usr/bin/env python
import torch
import numpy as np
import yaml
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_yaml(file_path):
    with open(file_path, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.SafeLoader)

def save_to_yaml(file_path, data):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

def print_params(params):
    print("=" * 20)
    print(f"Subject: {params['subj']}")
    print(f"Position: {params['train']['position']}")
    print(f"Velocity: {params['train']['velocity']}")
    print(f"Hidden Dim: {params['model']['hidden_dim']}")
    print(f"Preoutput Size: {params['model']['preoutput_size']}")
    print(f"Num Layers: {params['model']['num_layers']}")
    print(f"Dropout: {params['model']['dropout']}")
    print(f"Window Size: {params['window_size']}")
    print(f"Padding: {params['train']['padding']}")
    print("=" * 20)

def select_IO(params, data):
    """ Data initially was: 
    t, theta,phi,dtheta,dphi,rms1,rms2,rms3,rms4,nzc1,nzc2,nzc3,nzc4"
    """
    input_dim  = data.shape[-1]
    output_dim = 4

    if params['train']['emg_features'] == 'rms':
        data = np.delete(data,[-4,-3,-2,-1], axis=2)
        input_dim -= 4
    if params['train']['emg_features'] == 'nzc':
        data = np.delete(data,[-8,-7,-6,-5], axis=2)
        input_dim -= 4

    # select output
    if not params['train']['position']:
        data = np.delete(data, [0, 1], axis=2)
        input_dim -= params['dimension']
        output_dim -= params['dimension']

    if not params['train']['velocity']:
        data = np.delete(data, [2, 3], axis=2)
        input_dim -= params['dimension']
        output_dim -= params['dimension']

    if not params['model']['autoregressive']:
        input_dim -= output_dim

    return data, input_dim, output_dim

def select_IO_pendulum(params, data):
    """ Nota: the function is different for pendulum of EMG-imu data,
    because pendulum is: pos,vel,acc,u,udot (x10), and human does not have accel
    """
    input_dim  = data.shape[-1]
    output_dim = 6

    # select output
    if not params['train']['position']:
        data = np.delete(data, [0, 1], axis=2)
        input_dim -= params['dimension']
        output_dim -= params['dimension']

    if not params['train']['velocity']:
        data = np.delete(data, [2, 3], axis=2)
        input_dim -= params['dimension']
        output_dim -= params['dimension']
        if not params['train']['acceleration']:
            data = np.delete(data, [2, 3, 6, 7], axis=2)
            input_dim -= 4
            output_dim -= 2
    else:
        if not params['train']['acceleration']:
            data = np.delete(data, [4, 5, 8, 9], axis=2)
            input_dim -= 4
            output_dim -= 2

    if not params['model']['autoregressive']:
        input_dim -= output_dim

    return data, input_dim, output_dim

def check_data(t,data,output_dim,N,lim=None):
    fig, ax = plt.subplots(3,N, figsize=(18,5))
    for i in range(N):
        ax[0,i].plot(t,data[:,i,output_dim:]) #emg
        ax[1,i].plot(t,data[:,i,:output_dim]) #imu
        ax[2,i].plot(data[:,i,0],data[:,i,1]) #imu
        if lim is None:
            ax[2,i].set_xlim([-2,2])
            ax[2,i].set_ylim([-2,2])
        else:
            ax[2,i].set_xlim([-lim,lim])
            ax[2,i].set_ylim([-lim,lim])           
    ax[0,0].set_ylabel('EMG')
    ax[1,0].set_ylabel('IMU')
    ax[2,0].set_ylabel('Trajectory')
    plt.tight_layout()
    plt.show()

def normalize_data(data, output_dim, normalize_input, normalize_output,
                   save_folder, save=True,load=None):
    data_ = data.copy().reshape(data.shape[0] * data.shape[1], data.shape[2])

    if load == False:
        if normalize_input:
            mu_u, std_u = data_[:,output_dim:].mean(0), data_[:,output_dim:].std(0)
            data_[:, output_dim:] = (data_[:,output_dim:] - mu_u) / std_u
        if normalize_output:
            mu_y, std_y = data_[:,:output_dim].mean(0), data_[:,:output_dim].std(0)
            data_[:, :output_dim] = (data_[:,:output_dim] - mu_y) / std_y
        if save:
            np.save(f"{save_folder}/mu_u", mu_u)
            np.save(f"{save_folder}/std_u", std_u)
            np.save(f"{save_folder}/mu_y", mu_y)
            np.save(f"{save_folder}/std_y", std_y)
    else:
        mu_u  = np.load(f"{save_folder}/mu_u.npy")
        std_u = np.load(f"{save_folder}/std_u.npy")
        mu_y  = np.load(f"{save_folder}/mu_y.npy")
        std_y = np.load(f"{save_folder}/std_y.npy")
        if normalize_input:
            data_[:,output_dim:] = (data_[:,output_dim:] - mu_u)/ std_u
        if normalize_output:
            data_[:,:output_dim] = (data_[:,:output_dim] - mu_y)/ std_y
    print(f"mu_u:{mu_u}, std_mu:{std_u}")
    print(f"mu_y:{mu_y}, std_y:{std_y}")
    return data_.reshape(data.shape[0], data.shape[1], data.shape[2])


def add_padding(data,padding,window_size,t,dt,traj_first):
    if traj_first:
        data = data.transpose(1,0,2)
    # Pad input
    pad_len = 0
    t_pad = t
    if padding > 0:
        pad_len = padding * window_size
        x_pad   = np.repeat(data[0][np.newaxis,:], pad_len, axis=0)
        data  = np.append(x_pad, data, axis=0)
        t_pad = np.append(np.arange(-pad_len,0) * dt,t)
    if traj_first:
        data = data.transpose(1,0,2)
    return t_pad, data, pad_len


def select_train_test(data,train_perc,random=True,train_idx=None,test_idx=None):
    total_nb_trajectories = data.shape[1]
    Ntrain = int(train_perc*total_nb_trajectories)
    traj_idx = np.arange(total_nb_trajectories)
    if random:
        np.random.shuffle(traj_idx)
    if len(train_idx)==0 or len(test_idx)==0:
        train_idx, test_idx = traj_idx[:Ntrain], traj_idx[Ntrain:]
    
    return data[:,train_idx,:], data[:,test_idx,:], train_idx, test_idx

def select_test_CMW(data, t, output_dim, nb_options=5):
    total_nb_trajectories = data.shape[1]
    nb_ch = data.shape[-1] - output_dim
    workload = np.array([]).reshape(0,nb_ch)
    for k in range(total_nb_trajectories):
        emg_k = data[:,k,output_dim:] #RMS
        workload = np.concatenate((workload,[np.trapz(emg_k, t, axis=0)]),axis=0)

    workload_ratio = np.zeros_like(workload)
    for ch in range(nb_ch):
        other_channels = [j for j in range(nb_ch) if j != ch]
        workload_ratio[:,ch] = workload[:,ch]/workload[:,other_channels].sum(1)

    idx_sort = np.array([np.argsort(workload_ratio[:,ch])[::-1][:nb_options] for ch in range(nb_ch)]).T
    
    fig, ax = plt.subplots(nb_options, nb_ch, figsize=(8,6))
    for i in range(nb_options):
        for ch in range(nb_ch):
            k = idx_sort[i,ch]
            emg_i = data[:,k,output_dim:]
            ax[i,ch].plot(t, emg_i)

    fig, ax = plt.subplots(nb_options, nb_ch, figsize=(8,6))
    for i in range(nb_options):
        for ch in range(nb_ch):
            k = idx_sort[i,ch]
            imu_i = data[:,k,:output_dim]
            ax[i,ch].plot(imu_i[:,0],imu_i[:,1]); ax[i,ch].set_xlim([-1,1]); ax[i,ch].set_ylim([-1,1])
    plt.show()

    return idx_sort

def plot_train_test(train_data,test_data,lim):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_trajectories(axes[0], train_data, title='Training Trajectories',lim=lim)
    plot_trajectories(axes[1], test_data, title='Testing Trajectories'  ,lim=lim)
    plt.show()

def save_train_test(save_folder,train_data,test_data,train_idx,test_idx):
    np.save(f"{save_folder}/train_idx", train_idx)
    np.save(f"{save_folder}/test_idx", test_idx)
    np.save(f"{save_folder}/train", train_data)
    np.save(f"{save_folder}/test", test_data)
    print("Training and Testing data were saved...", train_data.shape, test_data.shape)

def load_train_test(save_folder):
    """ Load processed data splits (it's not normalized) """
    train_data = np.load(f"{save_folder}train.npy")
    test_data  = np.load(f"{save_folder}test.npy")    
    test_idx   = np.load(f"{save_folder}test_idx.npy")
    try:
        train_idx  = np.load(f"{save_folder}train_idx.npy")
    except:
        traj_ids = np.arange(train_data.shape[1]+test_data.shape[1])
        train_idx = np.array(list(set(traj_ids) - set(test_idx)))
    return train_data, test_data, train_idx, test_idx

def preprocess_data_for_lstm(input_data, output_data, window_size, window_step, delay_samples, device):
    x_ = torch.from_numpy(input_data).float().to(device)
    train_x_per_traj = x_.unfold(0, window_size, window_step).permute(1,0,3,2)[:,:-1,:,:]
    train_x = train_x_per_traj.reshape(-1,window_size,input_data.shape[-1])

    y_ = torch.from_numpy(output_data[window_size::window_step]).float().to(device)
    train_y_per_traj = y_.permute(1,0,2)
    train_y = train_y_per_traj.reshape(-1,output_data.shape[-1])
    print(f"x:{train_x.shape}, y:{train_y.shape}")
    return train_x, train_y


def train_model(model, params, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=params['train']['learning_rate'], 
                                weight_decay=params['train']['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                            factor=0.5, patience=params['train']['patience'], 
                            threshold=1e-3, threshold_mode='rel', cooldown=0,
                            min_lr=0, eps=1e-8, verbose=True)
    loss_fun = torch.nn.MSELoss()
    loss_log = []

    # convergence criteria
    loss_tolerance = params['train']['loss_tolerance']
    prev_loss = float('inf')
    epochs = 0
    t0 = time.time()

    # while epochs <= params['train']['num_epochs']:
    for i in range(params['train']['num_epochs']):
        try:
            if params['model']['stateful']:
                sample_batch = next(iter(data_loader))
                batch_size = sample_batch[0].size(0)

                h0 = torch.zeros(params['model']['num_layers'], batch_size, params['model']['hidden_dim']).to(device)
                c0 = torch.zeros_like(h0)

            batch_loss = []

            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                if params['model']['stateful']:
                    pred, (h0,c0) = model(batch_x, h0, c0)
                    h0.detach(), c0.detach()
                else:
                    pred, _ = model(batch_x)
                loss = loss_fun(pred, batch_y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())


            epoch_loss = np.array(batch_loss).mean()
            loss_log.append(epoch_loss)

            if params['train']['dynamic_lr']:
                scheduler.step(epoch_loss)

            if np.abs(epoch_loss - prev_loss) <= loss_tolerance:
                print("Convergence achieved. Stopping training.")
                break
            
            prev_loss = epoch_loss
            epochs += 1

            if params['train']['verbose']: # and epochs % 100 == 0:
                print("Epoch ", epochs, ": ", epoch_loss)
                

        except KeyboardInterrupt:
            break

    return model, loss_log, epochs, time.time()-t0

def get_params_dict(params, input_dim,output_dim,delay_samples,nb_traj_train):
    # save parameters
    params_to_save = {'input_dim':input_dim, 'output_dim':output_dim,
                    'hidden_dim':params['model']['hidden_dim'],
                    'preoutput_size':params['model']['preoutput_size'],
                    'num_layers':params['model']['num_layers'],
                    'dropout': params['model']['dropout'],
                    'window_size':params['window_size'],
                    'window_step':params['window_step'],
                    'delay': delay_samples,
                    'normalize_input': params['train']['normalize_input'],
                    'normalize_output':params['train']['normalize_output'],
                    'num_trajectories':nb_traj_train,
                    'model_name': params['model']['name'],
                    'subj':params['subj'],
                    'date':params['date'],
                    'position':params['train']['position'],
                    'velocity':params['train']['velocity'],
                    'pad_len':params['train']['padding'],
                    'emg_features':params['train']['emg_features'],
                    'autoregressive':params['model']['autoregressive'],
                    'batch_size':params['train']['batch_size']
                    }
    
    return params_to_save
    

# THIS IS CORRECT
def predict_motion(data,model,output_dim,window_size,device,autoregress=True,y_init=None):
    """ Input data: N x nbtraj x input_dim; for one traj: N x input_dim 
        Starts with window_size of labels
    """
    # if is 2 dimensional (1 trajectory only)
    if len(data.shape) == 2:
        u = torch.from_numpy(data[:,output_dim:]).float().to(device) # emg_input
        y = torch.from_numpy(data[:,:output_dim]).float().to(device) # imu_input
        Ypred = y[:window_size,:]

        with torch.no_grad():
            for i in range(window_size,len(data)):
                if autoregress:
                    y_new = Ypred[-window_size:]
                else:
                    y_new = y[i-window_size:i,:]
                
                x_net = torch.cat((y_new,u[i-window_size:i,:]),dim=-1).unsqueeze(0)
                ypred, _ = model(x_net) # (1,4)
                Ypred = torch.cat((Ypred,ypred),dim=0)

    if len(data.shape) == 3:
        u = torch.from_numpy(data[:,:,output_dim:]).float().to(device).permute(1,0,2) # emg_input
        y = torch.from_numpy(data[:,:,:output_dim]).float().to(device).permute(1,0,2) # imu_input
        Ypred = y[:,:window_size,:]

        with torch.no_grad():
            for i in range(window_size,len(data)):
                y_new = Ypred[:,-window_size:,:] if autoregress else y[:,i-window_size:i,:]
                x_net = torch.cat((y_new,u[:,i-window_size:i,:]),dim=-1)
                ypred, _ = model(x_net) # (1,4)
                Ypred = torch.cat((Ypred,ypred.unsqueeze(1)),dim=1)

    return Ypred.cpu().numpy()


def get_mse(ytrue,yhat):
    K, N, output_dim = ytrue.shape
    mse = np.zeros((K,N,output_dim))
    for k in range(K):
        for j in range(output_dim):
            mse[k,:,j] = abs(ytrue[k,:,j]-yhat[k,:,j])**2
    return mse


# def map_pred_to_screen(q,d1,d2,H,home_loc):
#     """ get trunk motion in radians, normalize by subject ROM and map it to screen"""
#     # normalize trunk angles (x and y between 0 and 1)
#     qx = np.where(q[:,0]<0, q[:,0]/H[0,0], q[:,0]/H[0,1])
#     qy = np.where(q[:,1]<0, q[:,1]/H[1,0], q[:,1]/H[1,1])

#     # now map to screen (qy negative bc of open cv img format)
#     px = qx * (d1/2)
#     py = -qy * (d2/2) if home_loc == "center" else qy * d2  

#     if home_loc == 'center':
#         offset = np.array([d1//2,d2//2])

#     p = np.array([px, py]).T + offset
#     return p

def map_pred_to_screen(q,d1,d2,H,home_loc,margin=None):
    """ get trunk motion in radians, normalize by subject ROM and map it to screen"""
    # normalize trunk angles (x and y between 0 and 1)
    qx = np.where(q[:,0]<0, q[:,0]/H[0,0], q[:,0]/H[0,1])
    qy = np.where(q[:,1]<0, q[:,1]/H[1,0], q[:,1]/H[1,1])

    # now map to screen (qy negative bc of open cv img format)
    margin = 0 if margin is None else margin
    px = qx * ((d1-margin)/2)
    py = -qy * ((d2-margin)/2) if home_loc == "center" else qy * (d2-margin)  

    if home_loc == 'center':
        offset = np.array([d1//2,d2//2])

    p = np.array([px, py]).T + offset
    return p


def plot_2Dtraj(ax, ytrue, ypred, title, colors):
    for traj in range(ytrue.shape[0]):
        ax.plot(ytrue[traj, :, 0], ytrue[traj, :, 1], color='k')
        ax.scatter(ytrue[traj, 0, 0], ytrue[traj, 0, 1], color='k')
        ax.scatter(ytrue[traj, -1, 0], ytrue[traj, -1, 1], color='maroon')
        # ax.plot(ypred[traj, :, 0], ypred[traj, :, 1], linestyle='--')
        # ax.scatter(ypred[traj, -1, 0], ypred[traj, -1, 1])
        # ax.scatter(ypred[traj, 0, 0], ypred[traj, 0, 1])
        ax.plot(ypred[traj, :, 0], ypred[traj, :, 1], color=colors[traj], linestyle='--')
        ax.scatter(ypred[traj, -1, 0], ypred[traj, -1, 1], color=colors[traj])
        ax.scatter(ypred[traj, 0, 0], ypred[traj, 0, 1], color=colors[traj])
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\phi$')
        ax.set_title(title)
        ax.grid(True)
    return ax

def plot_trajectories(ax,y,title,label_traj=False,lim=None):
    colors = sns.color_palette('hls', 1000)[::15]
    for i in range(y.shape[1]):
        if not label_traj:
            ax.plot(y[:,i,0], y[:,i,1], color=colors[i], linewidth=2, linestyle='-')
        else:
            ax.plot(y[:,i,0], y[:,i,1], color=colors[i], linestyle='-', linewidth=2,label=f'Traj{i+1}')
        ax.scatter(y[-1,i,0], y[-1,i,1], color=colors[i])
        ax.scatter(y[0,i,0], y[0,i,1], color='k')
        if label_traj:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_title(title)
    if lim is None:
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
    else:
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])       
    ax.grid()

    # plt.show()

def plot_reference(y, t, ref_emg, xlim=[-3.5,3.5],ylim=[-3.5,3.5], ytrue=None, ylim_emg=None):
    nb_traj_test = y.shape[0]
    nrows, ncols = 2, nb_traj_test
    c_base = ['blue', 'darkorange', 'green', 'red']
    muscles = ['RES','LES','ROE','LOE']

    fig, axes = plt.subplots(nrows,ncols,figsize=(14,nrows*3))
    for i, ax in enumerate(axes.flatten()):
        if i < nb_traj_test:
            if ytrue is not None:
                ax.scatter(ytrue[i,:,0],ytrue[i,:,1],s=1,color='k',label='Measured trajectory')
                ax.scatter(ytrue[i,0,0],ytrue[i,0,1],color='k')
                ax.scatter(ytrue[i,-1,0],ytrue[i,-1,1],color='k',s=15,marker='x')
            ax.scatter(y[i,:,0],y[i,:,1],s=1,color='r',label='Predicted trajectory')
            ax.scatter(y[i,0,0],y[i,0,1],s=30,color='r')
            ax.scatter(y[i,-1,0],y[i,-1,1],color='r',s=15,marker='x')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r'$p_{x}$')
            ax.set_title(f'Trajectory {i+1}') 
            ylabel = r'$p_{y}$'
        else:
            for j in range(ref_emg.shape[-1]):
                ax.plot(t,ref_emg[i-nb_traj_test,:,j],linewidth=1.5,label=muscles[j], color=c_base[j])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'EMG pattern {i-nb_traj_test+1}') 
            if ylim_emg is not None: ax.set_ylim(ylim_emg)
            ylabel = 'RMS of EMG (mV)'
        if i==0 or i==nb_traj_test:
            ax.set_ylabel(ylabel)
            ax.legend()
        ax.grid()
    plt.tight_layout()
    