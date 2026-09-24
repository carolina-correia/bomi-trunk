"""
This script provides tools to process and synchronize the 
3 signal sources
"""
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from scipy import signal
import pickle, copy
from scipy.spatial.transform import Rotation as R

DEBUG = 0
ROTATION_ORDER = 'xyz'

# Offline processing
def remove_peaks_plateaus(angle,thresh1, thresh2):
    # " This function removes peaks and plateaus based on thresholds"
    x = angle
    # " remove peaks "
    new_x = copy.copy(x)
    for i in range(1,len(x)-1):
        diffa, diffb = x[i] - x[i-1], x[i+1] - x[i]
        if np.abs(diffa) > thresh1 and np.abs(diffb) > thresh1:
            new_x[i] = new_x[i-1]
    # " correct plateaus "
    x_ = copy.copy(new_x)
    for i in range(0,len(new_x)-1):
        if np.abs(new_x[i+1] - x_[i]) > thresh2:
            x_[i+1] = x_[i]
        else:
            x_[i+1] = new_x[i+1]
    return x_


def get_IMU_torso_angles(euler,R_sw_init,thresh1,thresh2,kernel_size,B_imu,A_imu):
    eul_ = euler
    trunk_angles = np.array([]).reshape(0,3)
    for i in range(len(eul_)):
        R_sw = R.from_euler(ROTATION_ORDER, eul_[i,:], degrees=True).as_matrix() 
        R_diff = np.dot(R_sw_init.T,R_sw)
        euler_t = R.from_matrix(R_diff).as_euler(ROTATION_ORDER,degrees=True)
        trunk_angles = np.vstack((trunk_angles, euler_t))
    
    x = remove_peaks_plateaus(trunk_angles[:,0],thresh1[0],thresh2[0])
    y = remove_peaks_plateaus(trunk_angles[:,1],thresh1[1],thresh2[1])
    y = signal.medfilt(y, kernel_size)
    z = remove_peaks_plateaus(trunk_angles[:,2],thresh1[2],thresh2[2])
    trunk_motion = np.hstack((np.hstack((x[:,np.newaxis],y[:,np.newaxis])),z[:,np.newaxis]))

    if DEBUG:
        fig, ax = plt.subplots(3,1, figsize=(14, 7))
        ax[0].plot(trunk_angles[:,0], label='roll')
        ax[1].plot(trunk_angles[:,1], label='pitch')
        ax[2].plot(trunk_angles[:,2], label='yaw')
        ax[0].plot(trunk_motion[:,0], label='roll_filt', c='r')
        ax[1].plot(trunk_motion[:,1], label='pitch_filt', c='g')
        ax[2].plot(trunk_motion[:,2], label='yaw_filt', c='b')
        for i in range(3):
            ax[i].set_ylabel('Torso angle [degrees]')
            ax[i].legend()
        plt.show()

    eul_filt = signal.filtfilt(B_imu, A_imu,trunk_motion,axis=0)

    if DEBUG:
        fig, ax = plt.subplots(3,1, figsize=(14, 7))
        ax[0].plot(trunk_motion[:,0], label='roll')
        ax[0].plot(eul_filt[:,0], label='roll_filt', c='r')
        ax[1].plot(trunk_motion[:,1], label='pitch')
        ax[1].plot(eul_filt[:,1], label='pitch_filt', c='g')
        ax[2].plot(trunk_motion[:,2], label='yaw')
        ax[2].plot(eul_filt[:,2], label='yaw_filt', c='b')
        for i in range(3):
            ax[i].set_ylabel('Torso angle [degrees]')
            ax[i].legend()
        plt.show()

    return eul_filt



# nb 3, 6 and 7 are missing
def get_imuID(imu_nbs):
    for i in imu_nbs:
        if i == 1: IMU_id = '00B4341F'
        if i == 2: IMU_id = '00B4347F'
        if i == 3: IMU_id = '00B434 F' #TODO
        if i == 4: IMU_id = '00B4341B'
        if i == 5: IMU_id = '00B43429'          
        if i == 6: IMU_id = '00B434 ' #TODO
        if i == 7: IMU_id = '00B434 ' #TODO
        if i == 8: IMU_id = '00B43426'
    return IMU_id

def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                 np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

