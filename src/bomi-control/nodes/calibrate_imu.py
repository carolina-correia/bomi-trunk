#!/usr/bin/env python3
import os, sys
import time, datetime
import numpy as np
import pyautogui

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
    def __init__(self, calib_time, H_imu_path):
        rospy.init_node('calibration_node', anonymous=True)

        # params
        self.calib_time  = calib_time
        self.H_imu_path  = H_imu_path
        fc_imu, fs_imu = 15, 500
        self.B_imu,self.A_imu = signal.butter(7, fc_imu/(fs_imu/2), 'low') 

        # Subscribe to sensor data streaming topics
        self.emg_sub = rospy.Subscriber('/emg_features',Float64MultiArray,self.emg_listener)
        self.eul_sub = rospy.Subscriber('/imu_eulArray',Float64MultiArray, self.eul_listener)
        self.emgReceived = False
        self.eulReceived = False

        # Calibration
        self.trunk_left_imu, self.trunk_front_imu  = [], []
        self._pose_captured = False

        _rate = 120
        self.rate = rospy.Rate(_rate)

        time.sleep(5)
        self.check_calibration()

    def emg_listener(self,msg):
        self.features = msg.data
        self.emgReceived = True
 
    def eul_listener(self,msg):
        self.euler = msg.data
        self.eulReceived = True

    def check_calibration(self):
        if os.path.isfile(self.H_imu_path):
            self.H_imu = np.loadtxt(self.H_imu_path,delimiter=",")
            print("=====================================")
            print("Calibration files already exist!")            
            print("=====================================")
        else: 
            print("[Calibration] Calibration will now start!")
            self.calibrate_imu()


    def calibrate_imu(self):
        start_timer, start_timer2 = True, True
        pose_captured = False
        calib_completed = False

        while not rospy.is_shutdown() and not calib_completed:
            if self.eulReceived:
                R_imu = R.from_euler('xyz', self.euler, degrees=True).as_matrix() 

                if not pose_captured:
                    if start_timer:
                        print('[Calibration] Trunk detected. Do not move!')
                        t_start = time.time()
                        start_timer = False

                    if not pose_captured and time.time() - t_start > 5:
                        R_init_imu = R_imu
                        print("[Calibration] Caught initial pose")
                        pose_captured = True

                if pose_captured:
                    R_diff = np.dot(R_init_imu.T,R_imu)
                    eul_torso = R.from_matrix(R_diff).as_euler('xyz',degrees=True)
                    torso_left_imu, torso_front_imu = -eul_torso[2], -eul_torso[1]

                    if start_timer2:
                        time.sleep(2)
                        print("START MOVING !")
                        print("... torso right, left, front and repeat")
                        t_start = time.time()
                        start_timer2 = False
                    # print(time.time() - t_start)

                    if time.time() - t_start < self.calib_time:
                        self.trunk_left_imu.append(torso_left_imu)
                        self.trunk_front_imu.append(torso_front_imu) 
                    else:
                        print("Calibration COMPLETED !")
                        # IMU
                        plt.plot(self.trunk_left_imu, label='trunk_left_imu')
                        plt.plot(self.trunk_front_imu, label='trunk_front_imu')
                        plt.title('IMU')
                        plt.legend()
                        plt.show()
                        
                        ok = input("[Calibration] Calibration sucessfull? [y/n]")
                        if ok != 'y': raise Exception('[Calibration] Please repeat recalibration')                             
                        
                        self.save_calib()
                        print("[Calibration] Imu calib. matrix saved to:",self.H_imu_path)
                        calib_completed = True

            self.rate.sleep()

    def save_calib(self):
        theta, phi = -np.array(self.trunk_left_imu), np.array(self.trunk_front_imu)
        # symmetric calibration (sides only)
        min_theta, max_theta = abs(theta.min()), abs(theta.max())
        min_phi,   max_phi   = abs(phi.min()),   abs(phi.max())
        max_side = max(min_theta,max_theta)
        H = np.array([[max_side,max_side],
                        [min_phi,  max_phi]])
        np.save(self.H_imu_path,H)
        print("H:", H)

if __name__ == "__main__":

    params      = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    imu_params  = load_yaml(f"{bomicontrol_path}/cfg/imu.yaml")
    
    subj_day_folder = set_paths(params['record']['subj'],print_=1)
    H_imu_path  = f'{subj_day_folder}/interface/H_imu_deg'

    calibration = Sensor_Calibration(imu_params['calibration_time'], 
                                     H_imu_path)
