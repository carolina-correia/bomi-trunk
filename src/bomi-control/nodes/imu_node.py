#!/usr/bin/env python3
""" 
Class to retrieve euler angles in real-time from xsens IMU sensors
"""
import numpy as np
import rospy, yaml, rospkg
from hiros_xsens_mtw_wrapper.msg import Euler
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

class imuClient():
    def __init__(self,IMU_numbers,rate):

        rospy.init_node('imu_node', anonymous=True)
        print("[IMU_node] Starting acquisition")

        self.imu_nbs = IMU_numbers
        self.rate = rate
        self.nb_sensors = len(self.imu_nbs)

        # Re-publish but in array format
        self.eul_pub    = rospy.Publisher('/imu_eulArray',Float64MultiArray,    queue_size=30)
        self.quat_pub   = rospy.Publisher('/imu_quatArray',Float64MultiArray,   queue_size=30)
        self.linacc_pub = rospy.Publisher('/imu_linaccArray',Float64MultiArray, queue_size=30)
        self.angvel_pub = rospy.Publisher('/imu_angvelArray',Float64MultiArray, queue_size=30)
        self.eul_msg, self.quat_msg, self.linacc_msg, self.angvel_msg = Float64MultiArray(), Float64MultiArray(), Float64MultiArray(), Float64MultiArray()

        # Initialize empty arrays
        self.euler_angles = [np.zeros((1, 3), dtype=float) for _ in range(8)]
        self.quats        = [np.zeros((1, 4), dtype=float) for _ in range(8)]
        self.lin_accs     = [np.zeros((1, 3), dtype=float) for _ in range(8)]
        self.ang_vels     = [np.zeros((1, 3), dtype=float) for _ in range(8)]
        self.angles_received, self.data_received = False, False

        self.subscribe_IMUs()

    def subscribe_IMUs(self):
        # nb 3, 6 and 7 are missing!
        imu_ids = ['00B4341F','00B4347F','00B434?F','00B4341B','00B43429','00B434?','00B434?','00B43426']
        for i in self.imu_nbs:
            IMU_id = imu_ids[i-1]
            rospy.Subscriber(f'/xsens_node_01/{IMU_id}/filter/euler', Euler, lambda msg, idx=i-1: self.euler_listener(msg, idx))
            rospy.Subscriber(f'/xsens_node_01/{IMU_id}/imu/data', Imu, lambda msg, idx=i-1: self.imu_listener(msg, idx))

    # Callback functions:
    def euler_listener(self, msg, idx):
        self.euler_angles[idx] = np.zeros((1, 3), dtype=float)
        self.euler_angles[idx][0,0] = msg.roll
        self.euler_angles[idx][0,1] = msg.pitch
        self.euler_angles[idx][0,2] = msg.yaw
        self.angles_received = True

    def imu_listener(self, msg, idx):
        self.quats[idx] = np.zeros((1, 4), dtype=float)
        self.quats[idx][0,0] = msg.orientation.x
        self.quats[idx][0,1] = msg.orientation.y
        self.quats[idx][0,2] = msg.orientation.z
        self.quats[idx][0,3] = msg.orientation.w

        self.lin_accs[idx] = np.zeros((1, 3), dtype=float)
        self.lin_accs[idx][0,0] = msg.linear_acceleration.x
        self.lin_accs[idx][0,1] = msg.linear_acceleration.y
        self.lin_accs[idx][0,2] = msg.linear_acceleration.z

        self.ang_vels[idx] = np.zeros((1, 3), dtype=float)
        self.ang_vels[idx][0,0] = msg.angular_velocity.x
        self.ang_vels[idx][0,1] = msg.angular_velocity.y
        self.ang_vels[idx][0,2] = msg.angular_velocity.z
        self.data_received = True

    def get_array(self, data_list):
        all_data = np.array([], dtype=np.float64)
        for i in self.imu_nbs:
            all_data = np.append(all_data, data_list[i - 1])
        return all_data
    
    def run(self, eul, quat, linacc, angvel):
        rate = rospy.Rate(self.rate) #fs of 120Hz
        try:
            while not rospy.is_shutdown():
                if self.angles_received and self.data_received:
                    euler_t  = self.get_array(self.euler_angles)
                    quat_t   = self.get_array(self.quats)
                    linacc_t = self.get_array(self.lin_accs)
                    angvel_t = self.get_array(self.ang_vels)

                    self.eul_msg.data    = np.float64(euler_t).tolist()
                    self.quat_msg.data   = np.float64(quat_t).tolist()
                    self.linacc_msg.data = np.float64(linacc_t).tolist()
                    self.angvel_msg.data = np.float64(angvel_t).tolist()

                    if eul: self.eul_pub.publish(self.eul_msg)
                    if quat: self.quat_pub.publish(self.quat_msg)
                    if linacc: self.linacc_pub.publish(self.linacc_msg)
                    if angvel: self.angvel_pub.publish(self.angvel_msg)

                rate.sleep()

        except KeyboardInterrupt:
            print("[IMU_node] Stopped acquisition")

if __name__ == "__main__":

    bomi_path = rospkg.RosPack().get_path('bomicontrol')

    with open(bomi_path + "/cfg/imu.yaml", "r") as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.SafeLoader)

    imu_acquisition = imuClient(IMU_numbers=params['imu_nb'],rate=params['rate'])
    imu_acquisition.run(eul=1,quat=1,linacc=1,angvel=1)