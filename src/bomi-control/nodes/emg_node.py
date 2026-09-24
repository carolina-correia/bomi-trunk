#!/usr/bin/env python3
import os
import time, sys
import numpy as np
from scipy import signal

import rospy
from std_msgs.msg import Float64MultiArray

bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

# from sensors.emg.emgClient_Plus import emgClient
from sensors.emg.communication import *
from cfg.tools import load_yaml


class EMGnode:
    def __init__(self, params)-> None:

        rospy.init_node('emg_node', anonymous=True)
        print("[EMG_node] Starting acquisition")
        self.params = params

        # Publish emg data to topics
        self.emgPub          = rospy.Publisher('/emg_data',Float64MultiArray, queue_size=100)
        self.emgFiltPub      = rospy.Publisher('/emgFilt_data',Float64MultiArray, queue_size=100)
        self.featuresPub     = rospy.Publisher('/emg_features',Float64MultiArray, queue_size=50)
        self.featuresFiltPub = rospy.Publisher('/emg_featuresFilt',Float64MultiArray, queue_size=50)
        
        # initialize emg messages
        self.emg_msg          = Float64MultiArray()
        self.emgFilt_msg      = Float64MultiArray()
        self.features_msg     = Float64MultiArray()
        self.featuresFilt_msg = Float64MultiArray()

        # connect to EMG system
        serverIP = params['srvIP']
        port = params['port']     
        self.adaptor_chs = params['adaptor_chs']  
        self.device = params['device']

        if self.device == '64':
            self.mode, self.nch, self.fsamp = 1, 0, 2 #        
        if self.device == '64+':
            self.mode, self.nch, self.fsamp = 1, 0, 2 #   
        self.conn, self.nb_adapter_ch, _, self.bytes_per_ch = connect_to_sq(serverIP,port,self.mode,self.nch,self.fsamp)

        # filter emg data
        self.features_names = params['features']
        self.fs =  params['fs']
        bw = 0.5
        self.B_bp, self.A_bp = signal.butter(2,[params['fc_bp'][0]/(self.fs/2), params['fc_bp'][1]/(self.fs/2) ], 'band') 
        self.B_hp, self.A_hp = signal.butter(2, params['fc_high']/(self.fs/2), 'high') 
        self.B_s,  self.A_s  = signal.iirfilter(4, [(params['fc_stop']-bw)/(self.fs/2), (params['fc_stop']+bw)/(self.fs/2)], btype='bandstop', fs=self.fs)
        fs_features = round(params['twindow'] * self.fs) 

        # filter features
        self.Bf, self.Af = signal.butter(2, params['fc_features']/(self.fs/2), 'low') 
        self.filt_window = 150

        # get emg features and filter them
        self.emg_chs = params['emg_chs'] # 8
        self.channels_IDs = np.array(params['channels_IDs'])
        self.tw_samples     = round(params['twindow'] * self.fs)
        self.buffer_samples = round(params['tbuffer'] * self.fs)

        # Saving data
        self.nb_ch     = len(self.channels_IDs)
        self.nbfeat_ch = len(self.features_names)
        self.Features = np.array([], dtype=np.float32).reshape(0,self.nbfeat_ch * self.nb_ch)

        self.rate = rospy.Rate(self.fs)
        self.nb_samples = 2400
        

    def getSignals(self):
        if self.device == '64+':
            sample_from_channels_as_bytes = read_raw_bytes(self.nb_adapter_ch, self.bytes_per_ch, self.conn)
            sample_from_channels = bytes_to_integers(self.nb_adapter_ch, self.bytes_per_ch, sample_from_channels_as_bytes, output_milli_volts=True)
            return sample_from_channels

        if self.device == '64':
            tw = getSignals_old(self.conn,self.nb_samples,self.bytes_per_ch,self.adaptor_chs)
            return tw[-1,:]
    
    # publish features
    def run(self):
        isfirst = True
        last_tw = []
        emgdata = np.array([], dtype=np.float32).reshape(0,self.nb_ch) #all of them
        
        # features
        rms = lambda x: np.sqrt(np.mean(x**2,0))
        nzc = lambda x: ((x[:-1] * x[1:]) < 0).sum(0)
        last_time= time.time()

        try:
            while not rospy.is_shutdown():
                try:
                    emg_tw = self.getSignals() [:self.nb_ch] #Nx8
                except:
                    emg_tw = last_tw
                print(last_time-time.time())

                last_tw = emg_tw
                self.emg_msg.data = np.float64(last_tw).squeeze().tolist() #stream RAW EMG
                self.emgPub.publish(self.emg_msg)      # 1x1600 should be reshaped to 75x8    
                    
                if self.params['publish']['emg_filt']:
                    emgdata = np.vstack((emgdata,emg_tw))
                    if len(emgdata) >= self.tw_samples:
                        emg_data_tw = emgdata[-self.tw_samples:,:] #200x8                    
                        # self.emg_msg.data = np.float64(emg_data_tw[-1,:]).squeeze().tolist() #stream RAW EMG
                        # self.emgPub.publish(self.emg_msg)      # 1x1600 should be reshaped to 75x8    
                    
                        # Filter EMG data
                        if isfirst:
                            data_to_filt = emg_data_tw
                            isfirst = False
                        else:
                            data_to_filt = np.vstack((buffer,emg_data_tw))
                        
                        emgfilt_tw = signal.lfilter(self.B_bp, self.A_bp,data_to_filt, axis=0)
                        emgfilt_tw = signal.lfilter(self.B_hp, self.A_hp,emgfilt_tw, axis=0)
                        # emgfilt_tw = np.column_stack([np.convolve(emgfilt_tw[:,i], self.B_s/self.A_s, mode='same') for i in range(self.nb_ch)])

                        buffer = emgfilt_tw[-self.buffer_samples:,:]
                        self.emgFilt_msg.data = np.float64(emgfilt_tw[-1,:]).squeeze().tolist()
                        self.emgFiltPub.publish(self.emgFilt_msg)      # 1x1600 should be reshaped to 75x8    

                        # Extract features - RMS and NZC if selected
                        if self.params['publish']['emg_features_raw'] or self.params['publish']['emg_features_filt']:
                            features_tw = rms(emgfilt_tw)
                            if 'nzc' in self.features_names:
                                features_tw = np.append(features_tw, nzc(emgfilt_tw))
                            # print("AAAAA", self.Features.shape, features_tw.shape)

                            if self.params['publish']['emg_features_raw'] :                
                                self.features_msg.data = np.float64(features_tw).squeeze().tolist()
                                self.featuresPub.publish(self.features_msg)                             
                        
                            if self.params['publish']['emg_features_filt']:
                                # Filter the features:
                                self.Features = np.vstack((self.Features,features_tw))
                                if len(self.Features) >= self.filt_window:
                                    features_to_filt = self.Features[-self.filt_window:,:]
                                    features_filt_tw = signal.lfilter(self.Bf,self.Af,features_to_filt,axis=0) 
                                    self.featuresFilt_msg.data = np.float64(features_filt_tw[-1,:]).squeeze().tolist()
                                    self.featuresFiltPub.publish(self.featuresFilt_msg)   

                                if len(self.Features) >= 1000:
                                    self.Features = self.Features[-1000:,:]

                    if len(emgdata) >= self.tw_samples*3:
                        # print("[EMG_node]... Cleaning up EMG data stored")
                        emgdata = emgdata[-self.tw_samples:,:]


                if rospy.is_shutdown():
                    disconnect_from_sq(self.conn,self.mode,self.nch,self.fsamp)
                    break
                last_time= time.time()
                self.rate.sleep()

        except KeyboardInterrupt:
            print("[EMG_node] Shutting down ")
            return

if __name__ == "__main__":

    emg_params = load_yaml(f"{bomicontrol_path}/cfg/emg.yaml")
    print("[EMG_node] Params:", emg_params)

    emg_acquisition = EMGnode(emg_params)
    emg_acquisition.run()