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
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu
from hiros_xsens_mtw_wrapper.msg import Euler

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)
from cfg.tools import load_yaml, set_paths
from sensors.imu.tools import get_imuID

class SavingNode:
    def __init__(self, save_path, data_to_save, params, imu_params):

        rospy.init_node('savedata_node', anonymous=True)

        self.savepath = save_path
        self.data_to_save = data_to_save
        self.rate = params['record']['freq']
        self.nb_data_streams = len(data_to_save)
        self.control_mode    = params['record']['modality']

        # Subscribe to IMU data topics - for only 1 IMU
        if 'imu' in data_to_save:
            IMU_id = get_imuID(imu_params['imu_nb'])
            self.eul_sub = rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', Euler, self.euler_listener)
            self.imu_sub = rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', Imu, self.imu_listener)
        self.eulReceived, self.imudata_received = False, False
 
        # Subscribe to EMG data topics
        if 'emg' in data_to_save:
            self.emgdata_sub     = rospy.Subscriber('/emg_data',Float64MultiArray,self.emg_listener)
            self.emgfeatures_sub = rospy.Subscriber('/emg_featuresFilt',Float64MultiArray,self.emgfeatures_listener)
        self.emgReceived, self.featuresReceived = False, False

        # Subscribe to cursor interface topics
        if 'cursor' in data_to_save:
            self.cursorPos_sub = rospy.Subscriber('/cursor_pos',Pose2D, self.cursorPos_listener)
        self.cursorPosReceived = False

        if 'task' in data_to_save:
            self.trialID_sub   = rospy.Subscriber('/trial_ID',Int32,self.trial_ID_listener)
            self.trajID_sub    = rospy.Subscriber('/traj_ID',Int32,self.traj_ID_listener)
            
            self.ref_traj_pub     = rospy.Subscriber('/ref_trajectories',Float64MultiArray, self.ref_traj_listener)
            self.ref_time_pub     = rospy.Subscriber('/ref_time',Float64MultiArray, self.ref_time_listener)
            self.targets_list_pub = rospy.Subscriber('/targets_list',Float64MultiArray, self.targets_list_listener)

        self.trialIDReceived, self.trajIDReceived = False, False
        self.refTrajReceived, self.refTimeReceived,self.targetListReceived = False, False, False

        # Subscribe to interface topics with human commands
        if 'hmi' in data_to_save:
            self.torso_sub = rospy.Subscriber('/torso_angles',Pose2D,self.torsoangle_listener)
            self.cmd_sub   = rospy.Subscriber('/des_cmd',Pose2D,self.cmd_listener)
        self.cmdReceived, self.torsoReceived = False, False
        self.ypred_Received = False


        # Initialize data to save
        self.time  = []
        self.emgdata, self.emgfeatures = [], []
        self.eul, self.quat, self.linacc, self.angvel = [], [], [], []
        self.torso_angles = []
        self.desCmd = []
        self.cursorPos = []
        self.trialID = []
        self.trajID = []        
        self.emgPred = []
        self.curr_trial = 1
        self.ref_time = []
        self.trajectories = []
        self.targets_list = []
        self.rospy_rate = rospy.Rate(self.rate)

    def emg_listener(self,msg):
        self.emgdata_ = msg.data
        self.tw_samples = int(len(self.emgdata_)/8)
        self.emgReceived = True

    def emgfeatures_listener(self,msg):
        self.emgfeatures_ = msg.data
        self.nbfeatures = len(self.emgfeatures_)
        self.featuresReceived = True

    def euler_listener(self,msg):
        self.eul_ = np.zeros(3,dtype=float)
        self.eul_[0], self.eul_[1], self.eul_[2] = msg.roll, msg.pitch, msg.yaw
        self.eul_ = self.eul_.tolist()
        self.eulReceived = True

    def imu_listener(self,msg):
        self.quat_   = np.zeros(4,dtype=float)
        self.linacc_ = np.zeros(3,dtype=float)
        self.angvel_ = np.zeros(3,dtype=float)
        self.quat_[0],self.quat_[1],self.quat_[2],self.quat_[3] = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.linacc_[0], self.linacc_[1], self.linacc_[2] = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        self.angvel_[0], self.angvel_[1], self.angvel_[2] = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        self.quat_    = self.quat_.tolist()
        self.linacc_ = self.linacc_.tolist()
        self.angvel_ = self.angvel_.tolist()

        self.imudata_received = True

    def torsoangle_listener(self,msg):
        self.torso_angles_ = np.array([msg.x, msg.y])
        self.torsoReceived = True

    def cmd_listener(self,msg):
        self.cmd = np.array([msg.x, msg.y])
        self.cmdReceived = True

    def cursorPos_listener(self,msg):
        self.cursorPos_ = np.array([msg.x, msg.y])
        self.cursorPosReceived = True

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

    def targets_list_listener(self,msg):
        self.targets_list_ = msg.data
        self.targets_list = np.array(self.targets_list_).astype(int)
        self.targetListReceived = True

    def data_stream_counter(self):
        """ Check if all data has been received """
        streams_rcv_count = 0

        if 'imu' in data_to_save and self.eulReceived and self.imudata_received:
            streams_rcv_count +=1
            # print('IMU rcv')
        if 'emg' in data_to_save and self.emgReceived:
            streams_rcv_count +=1
            # print('EMG rcv')
        # if 'cursor' in data_to_save and self.cursorPosReceived and self.targetPosReceived and self.homePosReceived:
        if 'cursor' in data_to_save and self.cursorPosReceived:
            streams_rcv_count +=1
            # print("Cursor rcv")
        if 'hmi' in data_to_save and self.cmdReceived:
            streams_rcv_count +=1
            # print("HMI rsv")
        if 'task' in data_to_save and self.trialIDReceived:
            streams_rcv_count +=1
        
        if streams_rcv_count == self.nb_data_streams:
            return True
        else:
            return False

    def record(self):
        self.time.append(time.time())            

        if 'imu' in data_to_save:
            self.eul.append(self.eul_)
            self.quat.append(self.quat_)
            self.linacc.append(self.linacc_)
            self.angvel.append(self.angvel_)

        if 'emg' in data_to_save:
            self.emgdata.append(self.emgdata_)

        if self.featuresReceived:
            self.emgfeatures.append(self.emgfeatures_)
        
        if 'cursor' in data_to_save:
            self.cursorPos.append(self.cursorPos_)

        if 'task' in data_to_save:                
            self.trialID.append(self.trial_label)

        if self.trajIDReceived:
            self.trajID.append(self.traj_idx)

        if self.cmdReceived:
            self.desCmd.append(self.cmd)

        if self.torsoReceived:
            self.torso_angles.append(self.torso_angles_)

        if self.ypred_Received:
            self.emgPred.append(self.ypred_)
            

    def run(self):
        start_recording, _print, new_trial = False, True, True
        start_trial = False
        t0 = time.time()

        while not rospy.is_shutdown():

            start_recording = self.data_stream_counter()
            # print("BBBB", self.ref_time)

            if start_recording:
                if _print:
                    print("="*20+"\nSTART RECORDING\n"+"="*20)
                    _print = False  

                # Saving (trial IDs are coming from the Gui with the task)
                self.record()

            self.rospy_rate.sleep()

            if rospy.is_shutdown():
                alldata = {
                    'time': self.time,
                    'emgdata': self.emgdata,
                    'emgfeatures': self.emgfeatures,
                    'emgPred': self.emgPred,
                    'eul': self.eul,
                    'quat': self.quat,
                    'linacc': self.linacc,
                    'angvel': self.angvel,
                    'torso_angles': self.torso_angles,
                    'desCmd': self.desCmd,
                    'cursorPos': self.cursorPos,
                    'trialID': self.trialID,
                    'trajID': self.trajID,
                    'trajectories': self.trajectories,
                    'ref_time': self.ref_time,
                    'targets_list': self.targets_list,
                    'control_modality': self.control_mode
                }
                self.save(alldata)
                print('emgdata:',np.shape(np.array(self.emgdata)))
                print('time:',np.shape(np.array(self.time)))
                print('trialIDs:',len(self.trialID))
                break

    def save(self,data):
        f = open(self.savepath , "wb")
        pickle.dump(data, f)
        print("[SavingNode] Data saved to:", self.savepath)
        f.close()        


if __name__ == "__main__":

    params = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    imu_params = load_yaml(f"{bomicontrol_path}/cfg/imu.yaml")
    data_to_save = params['record']['data_to_save']

    subj_day_folder = set_paths(params['record']['subj'],print_=1)
    if params['interface']['task'] == 'path_following':
        folder = 'testing'
    else:
        folder = 'training'
    record_path = f"{subj_day_folder}/{folder}/{params['record']['modality']}"
    time_ = datetime.datetime.now().strftime("_%H_%M") 
    save_path_ = f'{record_path}/data{time_}.pkl'
    print("[SavingNode] Data to save:", save_path_)
    saving_data = SavingNode(save_path=save_path_, data_to_save=data_to_save, params=params, imu_params=imu_params)
    saving_data.run()
