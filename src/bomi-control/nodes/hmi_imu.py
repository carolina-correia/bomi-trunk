#!/usr/bin/env python3
" @author: Carolina Correia, @email:cgprcorreia@gmail.com "
"""
This node receives EMG data and computes the trunk angles in degrees.
It publishes the trunk angles and a scaled version, normalized by trunk ROM, if H is provided.
"""
import os, datetime, sys
import time
import numpy as np
import pyautogui

import rospy, yaml, rospkg
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32
from hiros_xsens_mtw_wrapper.msg import Euler

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from cfg.tools import get_subj_day_folder, load_yaml
from sensors.imu.tools import get_imuID

class IMU_interface:
    def __init__(self, imu_params,H_imu_path):

        rospy.init_node('imu_interface_node', anonymous=True)

        # Publish and subcribe to topics
        IMU_id = get_imuID(imu_params['imu_nb'])

        self.control_mode   = 'imu' #params['record']['modality']
        self.eul_sub        = rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', Euler, self.euler_listener)
        self.recalibIMU_sub = rospy.Subscriber('/recalibrate_IMU',Int32, self.recalib_listener)
        self.torso_pub      = rospy.Publisher('/torso_angles',Pose2D,queue_size=10)
        self.des_cmd        = rospy.Publisher('/des_cmd',Pose2D,queue_size=10)
        self.calibDone_pub  = rospy.Publisher('/calibrated',Int32, queue_size=1)

        # initialize arrays and messages
        self.trunk_angles   = np.array([]).reshape(0,2)
        self.eulReceived    = False
        self._pose_captured = False          
        self.recalibrate = 0
        self.calib_done  = 0
        self.input_torso = Pose2D()
        self.cmd         = Pose2D()
        self.calib_msg   = Int32()

        # Data processing params
        self.H_imu = np.load(H_imu_path)
        self.d1, self.d2 = pyautogui.size()
        self.filter_data = imu_params['filter']
        self.fs = imu_params['rate']
        self.B,self.A = signal.butter(7, imu_params['fc_lp']/(self.fs/2), 'low') 
        
        self.rate = rospy.Rate(self.fs)

    def euler_listener(self,msg):
        self.euler = np.zeros(3,dtype=float)
        self.euler[0] = msg.roll
        self.euler[1] = msg.pitch
        self.euler[2] = msg.yaw
        self.eulReceived = True

    def recalib_listener(self,msg):
        self.recalibrate = msg.data

    def publish_cmd(self,torso_right,torso_front,H):
        qx, qy = torso_right, torso_front
        s = np.array([qx, -qy])

        if self.filter_data: # this is a problem for the scaling
            self.trunk_angles = np.vstack((self.trunk_angles, s))
            if len(self.trunk_angles) > 19:
                qfilt = signal.lfilter(self.B,self.A,self.trunk_angles[-20:,:],axis=0)
                qx, qy = qfilt[-1,0], qfilt[-1,1]
        
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
            if self.eulReceived:
                # need to do some filtering on both of input signals!!
                R_trunk = R.from_euler('xyz', self.euler, degrees=True).as_matrix() 
                                
                if start_timer:
                    time.sleep(2)
                    print('[IMU_interface] Do not move!')
                    t_start = time.time()
                    start_timer = False
                if not self._pose_captured and time.time() - t_start > 2:
                    R_init = R_trunk
                    np.save(f"{subj_day_folder}interface/R_init.npy",R_init)
                    print("[IMU_interface] Caught initial pose")
                    self._pose_captured = True
                
                if self.recalibrate:
                    start_timer = True
                    self._pose_captured = False
                    self.calib_msg.data = 0
                    self.recalibrate = 0

                if self._pose_captured:
                    self.calib_msg.data = 1
                    R_diff = np.dot(R_init.T,R_trunk)
                    eul_torso = R.from_matrix(R_diff).as_euler('xyz',degrees=True)
                    torso_right, torso_front = eul_torso[2], -eul_torso[1]
                    self.publish_cmd(torso_right,torso_front,self.H_imu)
                
                self.calibDone_pub.publish(self.calib_msg)

            self.rate.sleep()


if __name__ == "__main__":

    test_params = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    imu_params  = load_yaml(f"{bomicontrol_path}/cfg/imu.yaml")
    print("[HMI_node] IMU params:", imu_params)

    # Set paths
    subj_day_folder = get_subj_day_folder(test_params['load']['subj'],test_params['load']['date'])
    H_imu_path  = f"{subj_day_folder}interface/H_imu_deg.npy"

    # run interface
    hmi = IMU_interface(imu_params,H_imu_path)
    hmi.run()