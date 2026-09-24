#!/usr/bin/env python
import os
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys, pickle, yaml, torch, datetime


# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)
from cfg.tools import *

class Data_processing():
    def __init__(self, data_path, emg_params, fs_recording, degrees, downsample_factor, debug=None, delay=None):
        self.data_path  = data_path
        self.emg_params = emg_params
        self.debug = debug
        self.emgdelay = delay
        # self.task = task

        self.ds_factor = downsample_factor if downsample_factor is not None else 1
        self.fs = fs_recording/self.ds_factor
        self.degrees = degrees

        self.emg_chs = emg_params['channels_IDs']
        # self.muscles = params['muscles_names']
        # self.colors  = ["blue","orange","g","r"]

        self.load_data()

    def load_data(self):        
        data = pickle.load(open(self.data_path,'rb'))
        print("Loading data:", self.data_path)

        self.t            = np.array(data['time'])         [::self.ds_factor]
        self.emgdata      = np.array(data['emgdata'])      [::self.ds_factor,self.emg_chs]
        # self.emgfeatures  = np.array(data['emgfeatures'])  [::self.ds_factor] # get RMS only
        self.eul          = np.array(data['eul'])          [::self.ds_factor] 
        self.quat         = np.array(data['quat'])         [::self.ds_factor]
        self.linacc       = np.array(data['linacc'])       [::self.ds_factor]
        self.angvel       = np.array(data['angvel'])       [::self.ds_factor]
        self.torso_angles = np.array(data['torso_angles']) [::self.ds_factor]
        self.desCmd       = np.array(data['desCmd'])       [::self.ds_factor]
        self.control_mode = data['control_modality']  
        try:
            self.trialID      = np.array(data['trialID'])      [::self.ds_factor]
            self.trajID       = np.array(data['trajID'])       [::self.ds_factor]
            self.cursorPos    = np.array(data['cursorPos'])    [::self.ds_factor]
            self.trajectories = np.array(data['trajectories'])    
            self.ref_time     = np.array(data['ref_time'])    
            self.targets_list = np.array(data['targets_list'])    
        except:
            pass

        self.t = self.t - self.t[0]
        self.eul_init    = self.eul[0,:]
        self.angvel_init = self.angvel[0,:]
        self.t_og = self.t
        self.emgdata_og = self.emgdata
        self.trialID_og = self.trialID
        
        # if self.emgdelay is not None:
        self.emgdata_og = correct_emg_delay(self.t_og,self.emgdata_og,time_delay=0.02)
        self.emgdata = self.emgdata_og

    def resample_data(self):

        t_resamp   = np.arange(0,self.t[-1],1/self.fs)
        self.emgdata      = np.array([np.interp(t_resamp, self.t, self.emgdata      [:,j]) for j in range(self.emgdata.shape[-1])]).T
        self.eul          = np.array([np.interp(t_resamp, self.t, self.eul          [:,j]) for j in range(self.eul.shape[-1])]).T 
        self.quat         = np.array([np.interp(t_resamp, self.t, self.quat         [:,j]) for j in range(self.quat.shape[-1])]).T
        self.linacc       = np.array([np.interp(t_resamp, self.t, self.linacc       [:,j]) for j in range(self.linacc.shape[-1])]).T
        self.angvel       = np.array([np.interp(t_resamp, self.t, self.angvel       [:,j]) for j in range(self.angvel.shape[-1])]).T
        self.torso_angles = np.array([np.interp(t_resamp, self.t, self.torso_angles [:,j]) for j in range(self.torso_angles.shape[-1])]).T
        self.desCmd       = np.array([np.interp(t_resamp, self.t, self.desCmd       [:,j]) for j in range(self.desCmd.shape[-1])]).T
        self.trialID      = np.round(np.interp(t_resamp, self.t, self.trialID)).astype(int)

        try:
            self.cursorPos    = np.array([np.interp(t_resamp, self.t, self.cursorPos [:,j]) for j in range(self.cursorPos.shape[-1])]).T
            self.trajID       = np.interp(t_resamp, self.t, self.trajID)
        except:
            pass

        self.t = t_resamp

    def get_torso_ang(self,eul0):
        """ 
        Receives euler angles in degrees, and returns torso angles in radians.
        Returns torso right flex angle and frontal flex angles
        """
        R_init  = R.from_euler('xyz', eul0, degrees=True).as_matrix()
        R_trunk = R.from_euler('xyz', self.eul, degrees=True).as_matrix()
        R_diff  = np.matmul(R_init.T, R_trunk)
        eul_torso = R.from_matrix(R_diff).as_euler('xyz', degrees=False)
        self.torso_angles = np.array([eul_torso[:, 2], -eul_torso[:, 1]]).T

    def get_torso_angvel(self, ang0):
        """ 
        Receives torso angular velocity in radians
        Returns torso right flex angle and frontal flex angular velocity (rad/s)"""
        R_init  = R.from_euler('xyz', ang0, degrees=False).as_matrix()
        R_trunk = R.from_euler('xyz', self.angvel, degrees=False).as_matrix()
        R_diff  = np.matmul(R_init.T, R_trunk)
        vel_torso = R.from_matrix(R_diff).as_euler('xyz', degrees=False)
        self.torso_vel = np.array([vel_torso[:, 2], -vel_torso[:, 1]]).T


    def crop_trials(self):

        trial_lengths = [len(self.trialID[self.trialID==i+1]) for i in range(self.nb_trials)]
        min_len = np.array(trial_lengths).min()
        self.nb_samples = min_len + int(self.fs)

        # exclude places where trialID == 0 (in between trials)
        idx_trials = np.array([])
        for i in range(self.nb_trials):
            idx_trial_i = np.where(self.trialID==i+1)[0]
            start_idx, end_idx = idx_trial_i[0] - int(self.fs), idx_trial_i[0] + min_len
            idx_trials = np.append(idx_trials, np.arange(start_idx,end_idx))

            # idx_trials = np.append(idx_trials,idx_trial_i[:min_len])
        idx_trials = idx_trials.astype(int)

        # Cut data
        self.t            = self.t            [idx_trials]
        self.emgdata      = self.emgdata      [idx_trials,:]
        self.eul          = self.eul          [idx_trials,:]
        self.angvel       = self.angvel       [idx_trials,:]        
        self.linacc       = self.linacc       [idx_trials,:]        
        try:
            self.desCmd   = self.desCmd       [idx_trials,:]
            self.cursorPos= self.cursorPos    [idx_trials,:]
            self.trajID   = self.trajID       [idx_trials]
        except:
            pass
        self.trialID  = self.trialID      [idx_trials]

    def reshape_data(self):
        self.t = self.t.reshape(self.nb_trials,self.nb_samples)
        self.emgdata = self.emgdata.reshape(self.nb_trials,self.nb_samples,self.emgdata.shape[-1])
        self.torso_angles = self.torso_angles.reshape(self.nb_trials,self.nb_samples,self.torso_angles.shape[-1])
        self.torso_vel    = self.torso_vel.reshape(self.nb_trials,self.nb_samples,self.torso_vel.shape[-1])
        
        try:
            self.desCmd       = self.desCmd.reshape(self.nb_trials,self.nb_samples,self.desCmd.shape[-1])
            self.cursorPos    = self.cursorPos.reshape(self.nb_trials,self.nb_samples,self.cursorPos.shape[-1])
            self.trajID       = self.trajID.reshape(self.nb_trials,self.nb_samples)
        except:
            pass

            
    def process_emg_imu(self):
        """ Data is already in shape (Nb traj, N, m)"""
        # filter design
        nb_ch   = self.emgdata.shape[-1]
        fs      = self.emg_params['fs']
        fc_bp   = self.emg_params['fc_bp'] #[30,400]
        fc_stop = self.emg_params['fc_stop']
        fc_high = self.emg_params['fc_high']
        bw_stop = 0.5
        B_bp, A_bp = signal.butter(2, [fc_bp[0]/(fs/2), fc_bp[1]/(fs/2)], 'band') 
        B_hp, A_hp = signal.butter(2, fc_high/(fs/2), 'high')
        B_s, A_s   = signal.iirfilter(4, [(fc_stop-bw_stop)/(fs/2), (fc_stop+bw_stop)/(fs/2)], btype='bandstop', fs=fs)

        # extract features offline
        rms = lambda x: torch.sqrt(torch.mean(x**2,dim=1))
        nzc = lambda x: ((x[:-1] * x[1:]) < 0).sum(0).unsqueeze(-1).T        
        
        window_size    = int(self.emg_params['twindow'] * fs)
        window_overlap = int(self.emg_params['tw_overlap'] * fs)
        window_step = window_size - window_overlap
        # print("window_size:",window_size)
        # print("window_step:",window_step)

        # filter features
        dt_features = window_step/fs
        fs_features = 1/dt_features # 10 Hz
        print("fs_features: ",fs_features)

        self.fs_features = fs_features
        fc_features = self.emg_params['fc_features'] # = 0.5 to keep ratio
        B_lp2, A_lp2 = signal.butter(2, fc_features/(fs_features/2), 'low')

        # filter velocities
        b, a = signal.butter(3, 0.8/(fs/2), 'low')
        self.torso_vel = signal.filtfilt(b,a,self.torso_vel,axis=1)

        # filter raw EMG
        emg_filt =  signal.filtfilt(B_bp, A_bp, self.emgdata, axis=1)
        emg_filt =  signal.filtfilt(B_hp, A_hp, emg_filt, axis=1)
        self.emg_filt = emg_filt
        # emg_filt = np.column_stack([np.convolve(emg_filt[:,:,i], B_s/A_s, mode='same') for i in range(nb_ch)])

        emgdata_filt = np.zeros_like((self.emgdata))
        N_features = int((self.nb_samples - window_size)/window_step)+1
        emgfeatures  = np.zeros((self.nb_trials,N_features,nb_ch*2))
        emgfeaturesFilt  = np.zeros((self.nb_trials,N_features,nb_ch*2))
        # print("emgfeatures",emgfeatures.shape)

        t_f = torch.zeros(self.nb_trials,N_features)
        torso_ang_f = torch.zeros(self.nb_trials,N_features,self.torso_angles.shape[-1])
        torso_vel_f = torch.zeros(self.nb_trials,N_features,self.torso_vel.shape[-1])
        cursorPos_f = torch.zeros(self.nb_trials,N_features,self.cursorPos.shape[-1])

        for k in range(self.nb_trials):
            # filter raw emg data
            # emg_filt_notch = np.column_stack([np.convolve(emg_filt[k,:,i], B_s/A_s, mode='same') for i in range(nb_ch)])
            # emgdata_filt[k,:,:] = emg_filt_notch

            # extract feaatures:
            # emgdata = torch.from_numpy(emg_filt_notch.copy())
            emgdata = torch.from_numpy(emg_filt[k,:,:].copy())
            emgdata = emgdata.unfold(0, window_size, window_step).permute(0,2,1)

            emgdata_rms  = rms(emgdata)
            emgdata_nzc  = torch.zeros(1,4)
            emgdata_nzc  = torch.cat([nzc(emgdata[i, :, :]) for i in range(len(emgdata))])
            emg_feat_k   = torch.cat((emgdata_rms,emgdata_nzc),dim=1).numpy()
            emgfeatures[k,:,:] = emg_feat_k
            emgfeaturesFilt[k,:,:] = signal.filtfilt(B_lp2, A_lp2,emg_feat_k,axis=0)
            
            t_f [k,:] = torch.from_numpy(self.t[k,:].copy()).unfold(0, window_size, window_step).mean(1)
            torso_ang_f[k,:,:]= torch.from_numpy(self.torso_angles[k,:,:].copy()).unfold(0, window_size, window_step).permute(0,2,1).mean(1)
            torso_vel_f[k,:,:]= torch.from_numpy(self.torso_vel[k,:,:].copy()).unfold(0, window_size, window_step).permute(0,2,1).mean(1)
            cursorPos_f[k,:,:]= torch.from_numpy(self.cursorPos[k,:,:].copy()).unfold(0, window_size, window_step).permute(0,2,1).mean(1)

        # resample (take the average in the time window)
        self.t_features            = t_f
        self.torso_angles_features = torso_ang_f 
        self.torso_vel_features    = torso_vel_f
        self.cursorPos_features    = cursorPos_f
    
        self.emgdataFilt           = emgdata_filt    
        self.emgfeatures           = emgfeatures    
        self.emgfeaturesFilt       = emgfeaturesFilt    

    def run(self):
        self.resample_data()

        self.nb_trials = len(np.unique(self.trialID)) -1
        self.crop_trials()

        # Get torso angles and angular velocity
        self.get_torso_ang(eul0=self.eul_init)
        self.get_torso_angvel(ang0=self.angvel_init)

        # Convert degrees to radians
        if self.degrees: 
            self.torso_vel = self.torso_vel * 180/np.pi

        # print('fs:',self.fs)
        # print('emgdata.shape',self.emgdata.shape)
        # print('nb trials:', self.nb_trials)


        self.reshape_data()
        # print('t.shape',self.t.shape)
        # print('emgdata.shape',self.emgdata.shape)

        self.process_emg_imu()

        # if self.debug:
        #     plt.subplots(figsize=(6,3))
        #     plt.plot(emg_features[:,:4])
        #     # plt.subplots(figsize=(6,3))
        #     plt.plot(feats_filt[:,:4])
        #     # plt.plot(feats_norm[:,:])

        # Visualize
        # if self.debug:
        #     plt.subplots(figsize=(6,3))
        #     plt.plot(self.t,(self.emgfeatures[:,:4]-self.emgfeatures[:,:4].mean(0))/self.emgfeatures[:,:4].std(0),label=['RES','LES','ROE','LOE'])
        #     plt.plot(self.t,(self.torso_angles-self.torso_angles.mean(0))/self.torso_angles.std(0),label=['theta','phi'])
        #     plt.plot(self.t,(self.torso_vel-self.torso_vel.mean(0))/self.torso_vel.std(0),label=['thetadot','phidot'])
        #     # plt.plot(self.t,self.torso_vel/np.max(abs(self.torso_vel),0))
        #     # plt.plot(self.t,self.emg_features/np.max(abs(self.emg_features),0))
        #     plt.plot(self.t,self.trialID/np.max(self.trialID),'k')
        #     plt.legend()
        #     plt.show()
        

        # Build dataset by trajectory
        pos_vel     = np.concatenate((self.torso_angles_features,self.torso_vel_features),axis=-1)
        pos_vel_emg = np.concatenate((pos_vel,self.emgfeaturesFilt),axis=-1)
        dataset     = np.concatenate((self.t_features[:,:,np.newaxis],pos_vel_emg),axis=-1)
        print("dataset:",dataset.shape)

        self.dataset = dataset
        return dataset
    
    def save(self, save_path):
        print('Dataset shape:',self.dataset.shape)
        np.save(save_path, self.dataset)
        print("Processed data saved to:",save_path)


def correct_emg_delay(t,emgdata,time_delay):
    nb_ch = emgdata.shape[-1]
    end_ = t[-1]
    start_time, end_time = 0 , end_ # seconds
    start_delay, end_delay = 0, -end_*time_delay  # seconds 0.02s delay EMG per second recorded

    delay_vector = np.interp(t, [start_time, end_time], [start_delay, end_delay])
    emg_nodelay = np.array([np.interp(t - delay_vector, t, emgdata[:,ch]) for ch in range(nb_ch)]).T
    return emg_nodelay


def extract_features(emgdata_i,time_i,params):
    fs      = params['emg']['fs']
    fc_bp   = params['emg']['fc_bp'] #[30,400]
    fc_stop = params['emg']['fc_stop']
    bw = 0.5
    B_bp, A_bp   = signal.butter(2, [fc_bp[0]/(fs/2), fc_bp[1]/(fs/2)], 'band') 
    B_hp, A_hp   = signal.butter(2, params['emg']['fc_high']/(fs/2), 'high')
    B_s, A_s   = signal.iirfilter(4, [(fc_stop-bw)/(fs/2), (fc_stop+bw)/(fs/2)], btype='bandstop', fs=fs)

    # filter raw emg data
    emgdata_filt =  signal.filtfilt(B_bp, A_bp, emgdata_i, axis=0)
    emgdata_filt =  signal.filtfilt(B_hp, A_hp,emgdata_filt,axis=0)
    emgdata_filt = np.column_stack([np.convolve(emgdata_filt[:,i], B_s/A_s, mode='same') for i in range(4)])

    window_size = int(params['emg']['twindow'] * fs)
    window_overlap = int(params['emg']['tw_overlap'] * fs)
    window_step = window_size - window_overlap
    emgdata = torch.from_numpy(emgdata_filt.copy())
    emgdata = emgdata.unfold(0, window_size, window_step).permute(0,2,1)

    rms = lambda x: torch.sqrt(torch.mean(x**2,dim=1))
    nzc = lambda x: ((x[:-1] * x[1:]) < 0).sum(0).unsqueeze(-1).T
    emgdata_rms  = rms(emgdata)
    emgdata_nzc  = torch.zeros(1,4)
    emgdata_nzc  = torch.cat([nzc(emgdata[i, :, :]) for i in range(len(emgdata))])
    emg_features = torch.cat((emgdata_rms,emgdata_nzc),dim=1)
    t_features = time_i[window_size::window_step]

    dt_features = np.mean(np.diff(t_features))
    fs_features = 1/dt_features # 20 Hz
    fc_features = params['emg']['fc_features'] # = 0.5 to keep ratio
    B_lp2, A_lp2 = signal.butter(2, fc_features/(fs_features/2), 'low')
    feats_filt = signal.filtfilt(B_lp2, A_lp2,emg_features.numpy(),axis=0)

    return t_features, feats_filt

def combine_datasets(data_path,files):
    alldata_ = []
    for i in range(len(files)):
        file_path = f"{data_path}imuemg_{files[i]}.npy"
        data_loaded = np.load(file_path)
        alldata_.append(data_loaded)
    alldata =  np.concatenate(alldata_, axis=1)
    time_ = datetime.datetime.now().strftime("_%H_%M")
    save_path = f"{data_path}imuemg{time_}"
    np.save(save_path, alldata)
    print("combined data saved as:",save_path)
    return alldata
