#!/usr/bin/env python3
" @author: Carolina Correia, @email:cgprcorreia@gmail.com "
"""
This node receives EMG data and computes the trunk angles in degrees.
It publishes the trunk angles and a scaled version, normalized by trunk ROM, if H is provided.
"""
import os, datetime, sys
import time, torch
import numpy as np
import pyautogui

import rospy, yaml, rospkg
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32
from hiros_xsens_mtw_wrapper.msg import Euler
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from cfg.tools import get_subj_day_folder, load_yaml
from sensors.imu.tools import get_imuID
from emg_regression.utils.torch_helper import TorchHelper
from emg_regression.approximators.lstm import LSTM


class EMG_interface:
    def __init__(self, subj_day_folder, imu_params):

        rospy.init_node('emg_interface_node', anonymous=True)

        H_imu_path  = f"{subj_day_folder}interface/H_imu_deg.npy"
        model_params = load_yaml(f"{subj_day_folder}model/lstm_config.yaml")
        self.model_path = f"{subj_day_folder}model/{model_params['model_name']}"

        self.mu_u  = np.load(f"{subj_day_folder}training/mu_u.npy")
        self.std_u = np.load(f"{subj_day_folder}training/std_u.npy")
        self.mu_y  = np.load(f"{subj_day_folder}training/mu_y.npy")
        self.std_y = np.load(f"{subj_day_folder}training/std_y.npy")


        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = model_params['window_size']

        # Publish and subcribe to topics

        # Listen to imu
        IMU_id = get_imuID(imu_params['imu_nb'])
        self.eul_sub = rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', Euler, self.euler_listener)
        self.imu_sub = rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', Imu, self.imu_listener)
        self.recalibIMU_sub = rospy.Subscriber('/recalibrate_IMU',Int32, self.recalib_listener)

        # Listen to EMG features
        self.emgfeatures_sub = rospy.Subscriber('/emg_featuresFilt',Float64MultiArray,self.emgfeatures_listener)
        self.featuresReceived = False

        self.control_mode   = 'imu' #params['record']['modality']

        self.des_cmd        = rospy.Publisher('/des_cmd',Pose2D,queue_size=10)
        self.calibDone_pub  = rospy.Publisher('/calibrated',Int32, queue_size=1)

        # initialize arrays and messages
        self.eulReceived    = False
        self.angvelReceived = False
        self._pose_captured = False          
        self.recalibrate = 0
        self.calib_done  = 0
        self.input_torso = Pose2D()
        self.cmd         = Pose2D()
        self.calib_msg   = Int32()

        self.trunk_data = np.array([]).reshape(0,model_params['output_dim'])
        self.input_data = np.array([]).reshape(0,model_params['input_dim'])

        # Data processing params
        self.H_imu = np.load(H_imu_path)
        self.d1, self.d2 = pyautogui.size()
        self.filter_data = imu_params['filter']
        self.fs = 2000
        self.B,self.A = signal.butter(7, imu_params['fc_lp']/(self.fs/2), 'low') 
        
        self.rate = rospy.Rate(self.fs)

        self.load_model()

    def load_model(self):
        self.model = LSTM(input_size=self.model_params['input_dim'], 
                hidden_dim=self.model_params['hidden_dim'], 
                pre_output_size=self.model_params['preoutput_size'],
                output_size=self.model_params['output_dim'], 
                dropout=self.model_params['dropout'],
                n_layers=self.model_params['num_layers']).to(self.device)

        TorchHelper.load(self.model, self.model_path, self.device)
        print('Model loaded:',self.model_path)
        self.model.eval()


    def euler_listener(self,msg):
        self.euler = np.zeros(3,dtype=float)
        self.euler[0] = msg.roll
        self.euler[1] = msg.pitch
        self.euler[2] = msg.yaw
        self.eulReceived = True

    def imu_listener(self,msg):
        self.angvel_ = np.zeros(3,dtype=float)
        self.angvel_[0], self.angvel_[1], self.angvel_[2] = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        self.angvel_ = self.angvel_.tolist()
        self.angvelReceived = True

    def emgfeatures_listener(self,msg):
        self.emgfeatures_ = np.array([msg.data])
        self.nbfeatures = len(self.emgfeatures_)
        self.featuresReceived = True


    def get_torso_posvel(self,R_init_eul, R_init_vel):
        """ Returns torso angular pos and vel in radians(/s)"""

        R_trunk_eul = R.from_euler('xyz', self.euler, degrees=True).as_matrix() 
        R_diff_eul = np.dot(R_init_eul.T,R_trunk_eul)
        eul_torso = R.from_matrix(R_diff_eul).as_euler('xyz',degrees=False)
        theta, phi = eul_torso[2], -eul_torso[1] # torso right and torso front

        R_trunk_vel = R.from_euler('xyz', self.angvel_, degrees=False).as_matrix()
        R_diff_vel  = np.matmul(R_init_vel.T, R_trunk_vel)
        vel_torso = R.from_matrix(R_diff_vel).as_euler('xyz', degrees=False)
        theta_dot, phi_dot = vel_torso[2], -vel_torso[1]

        self.trunk_data_ = np.array([[theta, phi, theta_dot, phi_dot]])

    def recalib_listener(self,msg):
        self.recalibrate = msg.data

    def publish_cmd(self,torso_right,torso_front,H):
        qx, qy = torso_right, torso_front
        s = np.array([qx, -qy])

        # raw angles side/front
        self.input_torso.x, self.input_torso.y = qx, qy
        self.torso_pub.publish(self.input_torso) 

        # to avoid shifts in position due to small displacements    
        if abs(qx) < 0.5: qx = 0
        if abs(qy) < 0.5: qy = 0

        # normalize trunk angles
        if qx < 0:  qx = qx/H[0,0] #maxL
        if qx >= 0: qx = qx/H[0,1] #maxR

        if qy >= 0: qy = qy/H[1,1] #maxF
        if qy < 0:  qy = qy/H[1,0] #maxB
        # qy = qy/H[1,1] #maxR    

        self.cmd.x, self.cmd.y = qx, qy
        self.des_cmd.publish(self.cmd) 

    def run(self):
        start_timer = True
        print("Starting IMU control")

        while not rospy.is_shutdown():
            if self.eulReceived and self.angvelReceived and self.featuresReceived:

                if start_timer:
                    time.sleep(2)
                    print('[IMU_interface] Do not move!')
                    t_start = time.time()
                    start_timer = False

                if not self._pose_captured and time.time() - t_start > 2:
                    R_init_eul = R.from_euler('xyz', self.euler, degrees=True).as_matrix() 
                    R_init_vel = R.from_euler('xyz', self.angvel_, degrees=False).as_matrix() 
                    # np.save(f"{subj_day_folder}interface/R_init.npy",R_init)
                    print("[IMU_interface] Caught initial pose")
                    self._pose_captured = True

                if self.recalibrate:
                    start_timer = True
                    self._pose_captured = False
                    self.calib_msg.data = 0
                    self.recalibrate = 0

                if self._pose_captured:
                    self.calib_msg.data = 1

                    self.get_torso_posvel(R_init_eul, R_init_vel)

                    # self.trunk_data = np.concatenate((self.trunk_data,self.trunk_data_),axis=0)

                    # normalize data
                    if self.model_params['normalize_input']:
                        self.trunk_data_  = (self.trunk_data_-self.mu_y)/self.std_y
                        self.emgfeatures_ = (self.emgfeatures_-self.mu_u)/self.std_u

                    # build input data imu + emg
                    input_data_ = np.concatenate((self.trunk_data_,self.emgfeatures_),axis=1)

                    self.input_data = np.concatenate((self.input_data,input_data_),axis=0)

                    if len(self.input_data) >= self.window_size:
                        x_tw = torch.from_numpy(self.input_data[-self.window_size:,:][np.newaxis,:,:]).float().to(self.device)

                        with torch.no_grad():
                            Ypred,_ = self.model(x_tw) # shape all traj, output
                    
                        if self.model_params['normalize_output']:
                            trunk_pred = Ypred.cpu().numpy() * self.std_y + self.mu_y
                            
                        print(trunk_pred*180/np.pi)


                    # print(input_data_)
                    # print(self.trunk_data.shape)

                    # self.publish_cursor_cmd(self.H_imu)
                
                self.calibDone_pub.publish(self.calib_msg)

            self.rate.sleep()


if __name__ == "__main__":

    test_params = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    imu_params  = load_yaml(f"{bomicontrol_path}/cfg/imu.yaml")
    print("[HMI_node] IMU params:", imu_params)

    # Set paths
    subj_day_folder = get_subj_day_folder(test_params['load']['subj'],test_params['load']['date'])

    # run interface
    hmi = EMG_interface(subj_day_folder, imu_params)
    hmi.run()