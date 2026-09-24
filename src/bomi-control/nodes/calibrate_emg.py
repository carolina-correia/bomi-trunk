#!/usr/bin/env python3
import os, sys
import time, datetime
import numpy as np
import pyautogui
import torch

import rospy, yaml, rospkg
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float64MultiArray

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from cfg.tools import set_paths, load_yaml

class Sensor_Calibration:
    def __init__(self, emg_params, H_emg_path):
        rospy.init_node('emg_calib_node', anonymous=True)

        # params
        self.calib_time  = 60
        self.H_emg_path  = H_emg_path

        self.nb_ch = 4
        self.muscles = ['Right ES','Left ES', 'Right OE','Left OE']
        self.colors = ['blue', 'orange', 'green', 'red']

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


        # listen to trial data recorded in real-time
        self.emgdata_sub = rospy.Subscriber('/emg_data',Float64MultiArray,self.emg_listener)
        self.emgReceived = False

        # Calibration
        self.time, self.emgdata, self.emgfeatures  = [], [], []
        self.trunk_left_imu, self.trunk_front_imu  = [], []
        self._pose_captured = False

 
        self.rate = fs
        self.rospy_rate = rospy.Rate(self.rate)

        print("[Calibration] Calibration will now start!")
        self.get_mvc()

    def emg_listener(self,msg):
        self.emgdata_ = msg.data
        self.emgReceived = True

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
        # plt.show()

        self.emgfeatures = signal.filtfilt(self.B_lp2,self.A_lp2,emgrms,axis=0)
        # self.emgfeatures = self.emgfeatures - self.emgfeatures[0,:]

        self.time = np.array(self.time)
        self.time -= self.time[0]
        self.time_f = self.time[self.window_size::self.window_step]


    def plot_emg(self):

        fig, ax = plt.subplots(self.nb_ch,1,figsize=(5,8))
        for ch in range(self.nb_ch):
            ax[ch].plot(self.time_f, self.emgfeatures[:,ch],linewidth=2,color=self.colors[ch])
            ax[ch].set_ylabel(f'EMG RMS (mV)')
            ax[ch].grid(); ax[ch].set_title(f'{self.muscles[ch]}')

            # Find the maximum value in the channel
            max_value = np.max(self.emgfeatures[:, ch])
            max_index = np.argmax(self.emgfeatures[:, ch])
            ax[ch].text(self.time_f[max_index], max_value, f'Max: {max_value:.2f}', color='k', fontsize=8,
                        horizontalalignment='right', verticalalignment='top')

        ax[-1].set_xlabel(f'Time (s)'); 
        plt.tight_layout()
        plt.show()

    def get_mvc(self):
        print1 = True
        calib_completed = False
        t_start = time.time()

        while not rospy.is_shutdown() and not calib_completed:
            if self.emgReceived:

                if print1:
                    print("[Calibration] START MVC")
                    print1 = False

                if time.time()-t_start <= self.calib_time:
                    self.record()
                    # print(len(self.emgdata))
                else:
                    print("Calibration COMPLETED !")
                    self.extract_features()
                    self.plot_emg()

                    ok = input("[Calibration] Calibration sucessfull? [y/n]")
                    if ok != 'y': raise Exception('[Calibration] Please repeat recalibration')                             
                    
                    self.save_calib()
                    print("[Calibration] EMG calib. matrix saved to:",self.H_emg_path)
                    calib_completed = True

            self.rospy_rate.sleep()

    def save_calib(self):
        mvc = self.emgfeatures.max(0)
        np.save(self.H_emg_path,mvc)
        print("H:", mvc)

if __name__ == "__main__":

    params      = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    emg_params  = load_yaml(f"{bomicontrol_path}/cfg/emg.yaml")
    
    subj_day_folder = set_paths(params['record']['subj'],print_=1)
    H_emg_path  = f'{subj_day_folder}/interface/emg_mvc'

    calibration = Sensor_Calibration(emg_params, 
                                     H_emg_path)
