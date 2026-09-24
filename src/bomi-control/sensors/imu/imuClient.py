#!/usr/bin/env python3
""" 
Class to retrieve euler angles in real-time from xsens IMU sensors
"""
import numpy as np
import rospy
from hiros_xsens_mtw_wrapper.msg import Euler
from sensor_msgs.msg import Imu

class imuClient():
    def __init__(self,IMU_numbers,rate):
        self.imu_nbs = IMU_numbers
        self.rate = rate
        self.nb_sensors = len(self.imu_nbs)
        
        self.euler_angles1 = np.zeros((1, 3),dtype=float)
        self.euler_angles2 = np.zeros((1, 3),dtype=float)
        self.euler_angles3 = np.zeros((1, 3),dtype=float)
        self.euler_angles4 = np.zeros((1, 3),dtype=float)
        self.euler_angles5 = np.zeros((1, 3),dtype=float)
        self.euler_angles6 = np.zeros((1, 3),dtype=float)
        self.euler_angles7 = np.zeros((1, 3),dtype=float)
        self.euler_angles8 = np.zeros((1, 3),dtype=float)
        self.angles_received = False

        self.quat_1 = np.zeros((1, 4),dtype=float)
        self.quat_2 = np.zeros((1, 4),dtype=float)
        self.quat_3 = np.zeros((1, 4),dtype=float)
        self.quat_4 = np.zeros((1, 4),dtype=float)
        self.quat_5 = np.zeros((1, 4),dtype=float)
        self.quat_6 = np.zeros((1, 4),dtype=float)
        self.quat_7 = np.zeros((1, 4),dtype=float)
        self.quat_8 = np.zeros((1, 4),dtype=float)
        
        self.lin_acc1 = np.zeros((1, 3),dtype=float)
        self.lin_acc2 = np.zeros((1, 3),dtype=float)
        self.lin_acc3 = np.zeros((1, 3),dtype=float)
        self.lin_acc4 = np.zeros((1, 3),dtype=float)
        self.lin_acc5 = np.zeros((1, 3),dtype=float)
        self.lin_acc6 = np.zeros((1, 3),dtype=float)
        self.lin_acc7 = np.zeros((1, 3),dtype=float)
        self.lin_acc8 = np.zeros((1, 3),dtype=float)

        self.ang_vel1 = np.zeros((1, 3),dtype=float)
        self.ang_vel2 = np.zeros((1, 3),dtype=float)
        self.ang_vel3 = np.zeros((1, 3),dtype=float)
        self.ang_vel4 = np.zeros((1, 3),dtype=float)
        self.ang_vel5 = np.zeros((1, 3),dtype=float)
        self.ang_vel6 = np.zeros((1, 3),dtype=float)
        self.ang_vel7 = np.zeros((1, 3),dtype=float)
        self.ang_vel8 = np.zeros((1, 3),dtype=float)
        self.data_received = False

        # print('--- IMU sensor numbers: ', IMU_numbers)
        # print('--- IMU rate: ', rate)

    "for IMUs from 1-8"
    # Callback functions:

    def euler_listener1(self,msg):
        self.euler_angles1 = np.zeros((1, 3),dtype=float)
        self.euler_angles1[0,0] = msg.roll
        self.euler_angles1[0,1] = msg.pitch
        self.euler_angles1[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener2(self,msg):
        self.euler_angles2 = np.zeros((1, 3),dtype=float)
        self.euler_angles2[0,0] = msg.roll
        self.euler_angles2[0,1] = msg.pitch
        self.euler_angles2[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener3(self,msg):
        self.euler_angles3 = np.zeros((1, 3),dtype=float)
        self.euler_angles3[0,0] = msg.roll
        self.euler_angles3[0,1] = msg.pitch
        self.euler_angles3[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener4(self,msg):
        self.euler_angles4 = np.zeros((1, 3),dtype=float)
        self.euler_angles4[0,0] = msg.roll
        self.euler_angles4[0,1] = msg.pitch
        self.euler_angles4[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener5(self,msg):
        self.euler_angles5 = np.zeros((1, 3),dtype=float)
        self.euler_angles5[0,0] = msg.roll
        self.euler_angles5[0,1] = msg.pitch
        self.euler_angles5[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener6(self,msg):
        euler_angles6 = np.zeros((1, 3),dtype=float)
        euler_angles6[0,0] = msg.roll
        euler_angles6[0,1] = msg.pitch
        euler_angles6[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener7(self,msg):
        self.euler_angles7 = np.zeros((1, 3),dtype=float)
        self.euler_angles7[0,0] = msg.roll
        self.euler_angles7[0,1] = msg.pitch
        self.euler_angles7[0,2] = msg.yaw
        self.angles_received = True

    def euler_listener8(self,msg):
        self.euler_angles8 = np.zeros((1, 3),dtype=float)
        self.euler_angles8[0,0] = msg.roll
        self.euler_angles8[0,1] = msg.pitch
        self.euler_angles8[0,2] = msg.yaw
        self.angles_received = True

    def imu_listener1(self,msg):
        self.quat_1 = np.zeros((1, 4),dtype=float)
        self.quat_1[0,0] = msg.orientation.x
        self.quat_1[0,1] = msg.orientation.y
        self.quat_1[0,2] = msg.orientation.z
        self.quat_1[0,3] = msg.orientation.w

        self.lin_acc1 = np.zeros((1, 3),dtype=float)
        self.lin_acc1[0,0] = msg.linear_acceleration.x
        self.lin_acc1[0,1] = msg.linear_acceleration.y
        self.lin_acc1[0,2] = msg.linear_acceleration.z

        self.ang_vel1 = np.zeros((1, 3),dtype=float)
        self.ang_vel1[0,0] = msg.angular_velocity.x
        self.ang_vel1[0,1] = msg.angular_velocity.y
        self.ang_vel1[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener2(self,msg):
        self.quat_2 = np.zeros((1, 4),dtype=float)
        self.quat_2[0,0] = msg.orientation.x
        self.quat_2[0,1] = msg.orientation.y
        self.quat_2[0,2] = msg.orientation.z
        self.quat_2[0,3] = msg.orientation.w

        self.lin_acc2 = np.zeros((1, 3),dtype=float)
        self.lin_acc2[0,0] = msg.linear_acceleration.x
        self.lin_acc2[0,1] = msg.linear_acceleration.y
        self.lin_acc2[0,2] = msg.linear_acceleration.z

        self.ang_vel2 = np.zeros((1, 3),dtype=float)
        self.ang_vel2[0,0] = msg.angular_velocity.x
        self.ang_vel2[0,1] = msg.angular_velocity.y
        self.ang_vel2[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener3(self,msg):
        self.quat_3 = np.zeros((1, 4),dtype=float)
        self.quat_3[0,0] = msg.orientation.x
        self.quat_3[0,1] = msg.orientation.y
        self.quat_3[0,2] = msg.orientation.z
        self.quat_3[0,3] = msg.orientation.w

        self.lin_acc3 = np.zeros((1, 3),dtype=float)
        self.lin_acc3[0,0] = msg.linear_acceleration.x
        self.lin_acc3[0,1] = msg.linear_acceleration.y
        self.lin_acc3[0,2] = msg.linear_acceleration.z

        self.ang_vel3 = np.zeros((1, 3),dtype=float)
        self.ang_vel3[0,0] = msg.angular_velocity.x
        self.ang_vel3[0,1] = msg.angular_velocity.y
        self.ang_vel3[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener4(self,msg):
        self.quat_4 = np.zeros((1, 4),dtype=float)
        self.quat_4[0,0] = msg.orientation.x
        self.quat_4[0,1] = msg.orientation.y
        self.quat_4[0,2] = msg.orientation.z
        self.quat_4[0,3] = msg.orientation.w

        self.lin_acc4 = np.zeros((1, 3),dtype=float)
        self.lin_acc4[0,0] = msg.linear_acceleration.x
        self.lin_acc4[0,1] = msg.linear_acceleration.y
        self.lin_acc4[0,2] = msg.linear_acceleration.z

        self.ang_vel4 = np.zeros((1, 3),dtype=float)
        self.ang_vel4[0,0] = msg.angular_velocity.x
        self.ang_vel4[0,1] = msg.angular_velocity.y
        self.ang_vel4[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener5(self,msg):
        self.quat_5 = np.zeros((1, 4),dtype=float)
        self.quat_5[0,0] = msg.orientation.x
        self.quat_5[0,1] = msg.orientation.y
        self.quat_5[0,2] = msg.orientation.z
        self.quat_5[0,3] = msg.orientation.w

        self.lin_acc5 = np.zeros((1, 3),dtype=float)
        self.lin_acc5[0,0] = msg.linear_acceleration.x
        self.lin_acc5[0,1] = msg.linear_acceleration.y
        self.lin_acc5[0,2] = msg.linear_acceleration.z

        self.ang_vel5 = np.zeros((1, 3),dtype=float)
        self.ang_vel5[0,0] = msg.angular_velocity.x
        self.ang_vel5[0,1] = msg.angular_velocity.y
        self.ang_vel5[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener6(self,msg):
        self.quat_6 = np.zeros((1, 4),dtype=float)
        self.quat_6[0,0] = msg.orientation.x
        self.quat_6[0,1] = msg.orientation.y
        self.quat_6[0,2] = msg.orientation.z
        self.quat_6[0,3] = msg.orientation.w

        self.lin_acc6 = np.zeros((1, 3),dtype=float)
        self.lin_acc6[0,0] = msg.linear_acceleration.x
        self.lin_acc6[0,1] = msg.linear_acceleration.y
        self.lin_acc6[0,2] = msg.linear_acceleration.z
        self.ang_vel6 = np.zeros((1, 3),dtype=float)
        self.ang_vel6[0,0] = msg.angular_velocity.x
        self.ang_vel6[0,1] = msg.angular_velocity.y
        self.ang_vel6[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener7(self,msg):
        self.quat_7 = np.zeros((1, 4),dtype=float)
        self.quat_7[0,0] = msg.orientation.x
        self.quat_7[0,1] = msg.orientation.y
        self.quat_7[0,2] = msg.orientation.z
        self.quat_7[0,3] = msg.orientation.w

        self.lin_acc7 = np.zeros((1, 3),dtype=float)
        self.lin_acc7[0,0] = msg.linear_acceleration.x
        self.lin_acc7[0,1] = msg.linear_acceleration.y
        self.lin_acc7[0,2] = msg.linear_acceleration.z
        self.ang_vel7 = np.zeros((1, 3),dtype=float)
        self.ang_vel7[0,0] = msg.angular_velocity.x
        self.ang_vel7[0,1] = msg.angular_velocity.y
        self.ang_vel7[0,2] = msg.angular_velocity.z
        self.data_received = True

    def imu_listener8(self,msg):
        self.quat_8 = np.zeros((1, 4),dtype=float)
        self.quat_8[0,0] = msg.orientation.x
        self.quat_8[0,1] = msg.orientation.y
        self.quat_8[0,2] = msg.orientation.z
        self.quat_8[0,3] = msg.orientation.w

        self.lin_acc8 = np.zeros((1, 3),dtype=float)
        self.lin_acc8[0,0] = msg.linear_acceleration.x
        self.lin_acc8[0,1] = msg.linear_acceleration.y
        self.lin_acc8[0,2] = msg.linear_acceleration.z
        self.ang_vel8 = np.zeros((1, 3),dtype=float)
        self.ang_vel8[0,0] = msg.angular_velocity.x
        self.ang_vel8[0,1] = msg.angular_velocity.y
        self.ang_vel8[0,2] = msg.angular_velocity.z
        self.data_received = True


    # nb 3, 6 and 7 are missing!
    def subscribe_IMUs(self):
        # create subscriber node
        # rospy.init_node('imu_node', anonymous=True)
        for i in self.imu_nbs:
            if i == 1:
                IMU_id = '00B4341F'
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener1)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener1)
            if i == 2:
                IMU_id = '00B4347F'
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener2)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener2)     
            if i == 3:
                IMU_id = '00B434 F' #TODO
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener3)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener3)
            if i == 4:
                IMU_id = '00B4341B'
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener4)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener4)  
            if i == 5:
                IMU_id = '00B43429'
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener5)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener5)              
            if i == 6:
                IMU_id = '00B434 ' #TODO
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener6)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener6)  
            if i == 7:
                IMU_id = '00B434 ' #TODO
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener7)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener7)                  
            if i == 8:
                IMU_id = '00B43426'
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/filter/euler', 
                Euler, self.euler_listener8)
                rospy.Subscriber('/xsens_node_01/'+ IMU_id+'/imu/data', 
                Imu, self.imu_listener8)  

    def get_angle_array(self):
        all_angles_t = np.array([],dtype=np.float64)
        for i in self.imu_nbs:
            if i == 1: all_angles_t = np.append(all_angles_t,self.euler_angles1)
            if i == 2: all_angles_t = np.append(all_angles_t,self.euler_angles2)
            if i == 3: all_angles_t = np.append(all_angles_t,self.euler_angles3)
            if i == 4: all_angles_t = np.append(all_angles_t,self.euler_angles4)
            if i == 5: all_angles_t = np.append(all_angles_t,self.euler_angles5)
            if i == 6: all_angles_t = np.append(all_angles_t,self.euler_angles6)
            if i == 7: all_angles_t = np.append(all_angles_t,self.euler_angles7)
            if i == 8: all_angles_t = np.append(all_angles_t,self.euler_angles8)
        return all_angles_t

    def get_quat_array(self):
        all_quat_t = np.array([],dtype=np.float64)
        for i in self.imu_nbs:
            if i == 1: all_quat_t = np.append(all_quat_t,self.quat_1)
            if i == 2: all_quat_t = np.append(all_quat_t,self.quat_2)
            if i == 3: all_quat_t = np.append(all_quat_t,self.quat_3)
            if i == 4: all_quat_t = np.append(all_quat_t,self.quat_4)
            if i == 5: all_quat_t = np.append(all_quat_t,self.quat_5)
            if i == 6: all_quat_t = np.append(all_quat_t,self.quat_6)
            if i == 7: all_quat_t = np.append(all_quat_t,self.quat_7)
            if i == 8: all_quat_t = np.append(all_quat_t,self.quat_8)
        return all_quat_t

    def get_linacc_array(self):
        all_linacc_t = np.array([],dtype=np.float64)
        for i in self.imu_nbs:
            if i == 1: all_linacc_t = np.append(all_linacc_t,self.lin_acc1)
            if i == 2: all_linacc_t = np.append(all_linacc_t,self.lin_acc2)
            if i == 3: all_linacc_t = np.append(all_linacc_t,self.lin_acc3)
            if i == 4: all_linacc_t = np.append(all_linacc_t,self.lin_acc4)
            if i == 5: all_linacc_t = np.append(all_linacc_t,self.lin_acc5)
            if i == 6: all_linacc_t = np.append(all_linacc_t,self.lin_acc6)
            if i == 7: all_linacc_t = np.append(all_linacc_t,self.lin_acc7)
            if i == 8: all_linacc_t = np.append(all_linacc_t,self.lin_acc8)
        return all_linacc_t

    def get_angvel_array(self):
        all_angvel_t = np.array([],dtype=np.float64)
        for i in self.imu_nbs:
            if i == 1: all_angvel_t = np.append(all_angvel_t,self.ang_vel1)
            if i == 2: all_angvel_t = np.append(all_angvel_t,self.ang_vel2)
            if i == 3: all_angvel_t = np.append(all_angvel_t,self.ang_vel3)
            if i == 4: all_angvel_t = np.append(all_angvel_t,self.ang_vel4)
            if i == 5: all_angvel_t = np.append(all_angvel_t,self.ang_vel5)
            if i == 6: all_angvel_t = np.append(all_angvel_t,self.ang_vel6)
            if i == 7: all_angvel_t = np.append(all_angvel_t,self.ang_vel7)
            if i == 8: all_angvel_t = np.append(all_angvel_t,self.ang_vel8)
        return all_angvel_t


    #TODO: this next function should not be called inside a loop    
    def getImuData(self):
        rospy.init_node('imu_node', anonymous=True)
        rate = rospy.Rate(self.rate) #fs of 120Hz
        self.subscribe_IMUs()

        while not rospy.is_shutdown():
            if self.angles_received and self.data_received:
                try:
                    euler_t = self.get_angle_array()    
                    quat_t = self.get_quat_array()
                    linacc_t = self.get_linacc_array()
                    angvel_t = self.get_angvel_array()  
    
                except KeyboardInterrupt:
                    break
                rate.sleep()
                return euler_t, quat_t, linacc_t, angvel_t

if __name__ == "__main__":
    # from cfg import *
    imu_acquisition = imuClient(IMU_numbers=[1],rate=120)
    while True:
        try:
            euler_t, quat_t, linacc_t, angvel_t =imu_acquisition.getImuData()
            print(euler_t)
        except:
            break

# IMUclient = imuClient(IMU_numbers=[2],rate=120) 
# # while not rospy.is_shutdown():
# #     try:
# #         euler_t, quat_t, linacc_t, angvel_t = IMUclient.getImuData()
# #         print(euler_t)
# #     except:
# #         pass
    

# IMUclient.subscribe_IMUs()
# rate = rospy.Rate(IMUclient.rate) #fs of 120Hz

# while not rospy.is_shutdown():
#     if IMUclient.angles_received and IMUclient.data_received:
#         # try:
#         euler_t  = IMUclient.get_angle_array()    
#         quat_t   = IMUclient.get_quat_array()
#         linacc_t = IMUclient.get_linacc_array()
#         angvel_t = IMUclient.get_angvel_array()   
#         print(euler_t)
          
#         # except KeyboardInterrupt:
#         #     break
#         rate.sleep()
#         # return euler_t, quat_t, linacc_t, angvel_t
