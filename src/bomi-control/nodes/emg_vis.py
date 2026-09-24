#!/usr/bin/env python3
"""
This script contains a Saving Node:
It simultaneously records data from all EMG, camera, IMU and cursor topics, 
whether or not data is being published to those topics. The data is saved to a file inside
the "TYPE" of recording (recording/ or task_/), with the current time so that files are never overwritten.
"""
import time, datetime
import numpy as np
import pickle, sys, os
import rospy
from std_msgs.msg import Float64MultiArray, Int32
import matplotlib.pyplot as plt
from scipy import signal
import torch

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)
from cfg.tools import *
from interface.tools import muscle_color_grad, normalize_time_series

class EMGvis_Node:
    def __init__(self, params, emg_params, t_ref, emg_ref,save_path,emg_mvc):

        rospy.init_node('emgvis_node', anonymous=True)

        self.t_ref   = t_ref
        self.t_ref_perc = normalize_time_series(self.t_ref)
        self.emg_ref = emg_ref
        self.mvc = emg_mvc
        self.nb_ch = self.emg_ref.shape[-1]
        self.muscles = ['Right ES','Left ES', 'Right OE','Left OE']
        self.colors = ['blue', 'orange', 'green', 'red']
        self.task = params['interface']['task']
        self.color_grad = muscle_color_grad(num_colors=55)
        self.savepath = save_path

        #filter EMG
        fs      = emg_params['fs']
        fc_bp   = emg_params['fc_bp'] #[30,400]
        fc_high = emg_params['fc_high']
        self.B_bp, self.A_bp = signal.butter(2, [fc_bp[0]/(fs/2), fc_bp[1]/(fs/2)], 'band') 
        self.B_hp, self.A_hp = signal.butter(2, fc_high/(fs/2), 'high')

        # extract and filter features
        self.window_size    = int(emg_params['twindow'] * fs)
        self.window_overlap = int(emg_params['tw_overlap'] * fs)
        self.window_step = self.window_size - self.window_overlap        
        dt_features = self.window_step/fs
        fs_features = 1/dt_features # 10 Hz
        fc_features = emg_params['fc_features'] # = 0.5 to keep ratio
        self.B_lp2, self.A_lp2 = signal.butter(2, fc_features/(fs_features/2), 'low')


        self.rate = params['record']['freq']

        # listen to trial data recorded in real-time
        self.emgdata_sub     = rospy.Subscriber('/emg_data',Float64MultiArray,self.emg_listener)
        self.emgfeatures_sub = rospy.Subscriber('/emg_featuresFilt',Float64MultiArray,self.emgfeatures_listener)

        self.trialID_sub   = rospy.Subscriber('/trial_ID',Int32,self.trial_ID_listener)
        self.trajID_sub    = rospy.Subscriber('/traj_ID',Int32,self.traj_ID_listener)
        
        self.ref_traj_pub  = rospy.Subscriber('/ref_trajectories',Float64MultiArray, self.ref_traj_listener)
        self.ref_time_pub  = rospy.Subscriber('/ref_time',Float64MultiArray, self.ref_time_listener)

        self.emgReceived, self.featuresReceived = False, False
        self.trialIDReceived, self.trajIDReceived = False, False
        self.refTrajReceived, self.refTimeReceived= False, False

        # Initialize data to save
        self.time, self.emgdata, self.emgfeatures  = [], [], []
        self.t_trials, self.emg_trials  = [], []
        self.t_trials_perc = []
        
        self.trialID = []
        self.trajID = []        
        self.curr_trial = 1
        self.rospy_rate = rospy.Rate(self.rate)

    def emg_listener(self,msg):
        self.emgdata_ = msg.data
        self.tw_samples = int(len(self.emgdata_)/8)
        self.emgReceived = True

    def emgfeatures_listener(self,msg):
        self.emgfeatures_ = msg.data
        self.nbfeatures = len(self.emgfeatures_)
        self.featuresReceived = True

    def trial_ID_listener(self,msg):
        self.trial_label = msg.data
        self.trialIDReceived = True
        
    def traj_ID_listener(self,msg):
        self.traj_idx = msg.data
        self.trajIDReceived = True

    def ref_time_listener(self,msg):
        self.ref_time = msg.data
        self.refTimeReceived = True
        
    def ref_traj_listener(self,msg):
        self.ref_traj_ = msg.data
        if self.refTimeReceived:
            nb_samples = len(self.ref_time)
            nb_traj = len(self.ref_traj_) // (nb_samples * 2)
            self.trajectories = np.array(self.ref_traj_).reshape(nb_traj,nb_samples,2)
        self.refTrajReceived = True

    def record(self):
        self.time.append(time.time())            
        self.emgdata.append(self.emgdata_)

    def extract_features(self):
        rms = lambda x: torch.sqrt(torch.mean(x**2,dim=1))
        emgdata = np.array(self.emgdata)
        emg_filt =  signal.filtfilt(self.B_bp, self.A_bp, emgdata, axis=0)
        emg_filt =  signal.filtfilt(self.B_hp, self.A_hp,emg_filt,axis=0)

        # extract feaatures:
        emgdata = torch.from_numpy(emg_filt.copy())
        emgdata = emgdata.unfold(0, self.window_size, self.window_step).permute(0,2,1)
        emgrms = rms(emgdata).numpy()
        self.emgfeatures = signal.filtfilt(self.B_lp2,self.A_lp2,emgrms,axis=0)
        # self.emgfeatures = self.emgfeatures - self.emgfeatures[0,:]

        self.time = np.array(self.time)
        self.time -= self.time[0]
        self.time_f = self.time[self.window_size::self.window_step]
        self.time_perc = normalize_time_series(self.time_f)
        # print("AAADSE",self.time_f.shape,self.time_perc.shape, self.emgfeatures.shape)

        # append to all trials
        self.t_trials_perc.append(self.time_perc) 
        self.t_trials.append(self.time_f) 
        self.emg_trials.append(self.emgfeatures/self.mvc)

    def plot_emg(self):
        if self.task == 'path_following':
            self.curr_emg_ref = self.emg_ref[self.traj_idx,:,:]

        fig, ax = plt.subplots(self.nb_ch,1,figsize=(5,6))
        for ch in range(self.nb_ch):
            if self.task == 'path_following':
                ax[ch].plot(self.t_ref_perc, self.curr_emg_ref[:,ch],linewidth=1.5,color='k')
                ax[ch].plot(self.time_perc, self.emgfeatures[:,ch],linewidth=1.5,color=self.colors[ch])

                # ax[ch].plot(self.t_ref_perc, self.curr_emg_ref[:,ch]/self.mvc[ch],linewidth=1.5,color='k')
                # ax[ch].plot(self.time_perc, self.emgfeatures[:,ch]/self.mvc[ch],linewidth=1.5,color=self.colors[ch])
            # else:
            # for i in range(len(self.emg_trials)):
            #     ax[ch].plot(self.t_trials_perc[i], self.emg_trials[i][:,ch],linewidth=2,color=self.color_grad[ch][i])

            # ax[ch].set_ylim([0,1])
            ax[ch].set_ylabel(f'EMG RMS (mV)')
            ax[ch].grid(); ax[ch].set_title(f'{self.muscles[ch]}')
        ax[-1].set_xlabel(f'Time (s)'); 
        plt.tight_layout()
        plt.show()
        

    def run(self):
        start_trial = False
        trial_labels = []

        while not rospy.is_shutdown():

            if self.trialIDReceived:
                trial_labels.append(self.trial_label)

                if len(trial_labels)>1:
                    if (trial_labels[-1] - trial_labels[-2]) > 0:
                        start_trial = True
                    if (trial_labels[-1] - trial_labels[-2]) < 0 and start_trial:
                        self.extract_features()
                        self.plot_emg()
                        start_trial = False
                    trial_labels = trial_labels[-2:]

                if start_trial:
                    self.record()

                else:
                    self.time  = []
                    self.emgdata = []


            self.rospy_rate.sleep()

            if rospy.is_shutdown():
                alldata = {'t': self.t_trials,
                           'emg':self.emg_trials}
                self.save(alldata)
                print("[EMGvisNode] shuttting down")
                print('emgdata:',np.shape(np.array(self.emgdata)))
                print('time:',np.shape(np.array(self.time)))
                break

    def save(self,data):
        f = open(self.savepath , "wb")
        pickle.dump(data, f)
        print("[SavingNode] Data saved to:", self.savepath)
        f.close()    

if __name__ == "__main__":

    params     = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    emg_params = load_yaml(f"{bomicontrol_path}/cfg/emg.yaml")

    subj_day_folder = get_subj_day_folder(params['load']['subj'],params['load']['date'])
    emg_ref = np.load(f"{subj_day_folder}testing/ref_emg.npy")
    t_ref   = np.load(f"{subj_day_folder}training/time.npy")
    t_ref  -= t_ref[0]

    emg_mvc = np.load(f"{subj_day_folder}interface/emg_mvc.npy")

    subj_day_folder_save = set_paths(params['record']['subj'],print_=1)
    folder = 'testing' if params['interface']['task'] == 'path_following' else 'training'
    time_ = datetime.datetime.now().strftime("_%H_%M") 
    save_path_ = f"{subj_day_folder_save}/{folder}/{params['record']['modality']}/emgfollow{time_}.pkl"

    emgvis = EMGvis_Node(params, emg_params, t_ref, emg_ref, save_path_, emg_mvc)
    emgvis.run()