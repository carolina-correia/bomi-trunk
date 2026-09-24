#!/usr/bin/env python3
" @author: Carolina Correia, @email:cgprcorreia@gmail.com "
"""
Cursor node: Receives a desired command (2 angles, between -1 and 1), and maps it to the screen
The cursor gains can be adjusted
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2, pyautogui
import sys, os

import rospy, yaml, rospkg
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int32

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from interface.displays import *
from interface.tools import *
from cfg.tools import load_yaml

# class for controlling cursor 
class Cursor_interface:
    def __init__(self, params)-> None:

        rospy.init_node('cursor_node', anonymous=True)

        # Interface params
        self.d1, self.d2 = pyautogui.size() # w, h
        self.home_loc = params['home_loc']
        self.r_cursor = params['radius']

        # Cursor controller parameters
        self.dt    = params['dt']     
        self.gains = params['gains']  
        self.controller_type =  params['controller']

        self.k        = params['stiffness']  
        self.d        = params['damping']  
        self.mass     = params['mass']  
        self.friction = params['friction']  
        self.A        = np.array(params['A'])  

        # Publish and subcribe to topics
        self.cmd_sub       = rospy.Subscriber('/des_cmd',Pose2D, self.cmd_listener)
        self.cursorPos_pub = rospy.Publisher('/cursor_pos',Pose2D, queue_size=1)
        self.recalibIMU_sub = rospy.Subscriber('/recalibrate_IMU',Int32, self.recalib_listener)
        self.recalibrate = 0

        # initialize arrays and messages
        self.cmdReceived = False
        self.calib_done  = False
        self.cursPos     = Pose2D()
        self.homePos     = Pose2D()

        # initialize target, trajectory and cursor
        self.initialize_cursor()

        self.t_start = 0
        self.freq = params['freq'] ### TODO
        self.rate = rospy.Rate(self.freq)

        self.force_field = np.dot(-0.05,(self.p_home - np.array([self.d1, self.d2//2])))

    def cmd_listener(self,msg):
        self.cmd = np.array([msg.x, msg.y])
        self.cmdReceived = True   

    def recalib_listener(self,msg):
        self.recalibrate = msg.data

    def initialize_cursor(self):
        self.cursor_pos = np.array(self.p_home)
        self.cursor_vel = 0
        self.cursor_acc = 0
          
    def move_cursor(self, targetPos):
        """ Receives scaled desCmd (qx,qy) between -1, 1, and maps it to screen """
        # Update cursor_position and bound it to screen limits
        if self.controller_type == 'DS':
            self.desVel     = np.dot(self.A,(np.array(self.cursor_pos) - targetPos))
            self.cursor_pos = self.cursor_pos + self.desVel*self.dt
            
        if self.controller_type == 'MSD':
            Fc = targetPos
            acc = (Fc - self.k*self.cursor_pos - self.d*vel)/self.mass
            vel = vel + acc * self.dt
            self.cursor_pos = self.cursor_pos + vel*self.dt
            
        if self.controller_type == 'PD':
            Fc = self.k * (targetPos-self.cursor_pos) - self.d * self.cursor_vel
            # acc = (Fc - self.friction * self.cursor_vel)/self.mass
            Fk = -0.5 * (self.p_home - self.cursor_pos)
            acc = (Fc + Fk)/self.mass
            self.cursor_vel = self.cursor_vel + self.dt * acc
            self.cursor_pos = self.cursor_pos + self.dt * self.cursor_vel

        self.cursor_pos = self.cursor_pos.astype(int)
        self.cursor_pos = bound_cursor(self.cursor_pos,self.r_cursor,self.d1,self.d2)


    def run(self):
        while not rospy.is_shutdown():
            try:
                if self.cmdReceived:
                    # get human position cmd [-1,1] and map to cursor position
                    q = np.array([self.cmd[0], self.cmd[1]])
                    desPos = map_body_to_cursor(q,self.d1,self.d2,self.home_loc,self.gains,self.p_home)
                    self.move_cursor(desPos)

                    # If imu recalibration, reset cursor position
                    if self.recalibrate:
                        self.initialize_cursor()

                    # Publish cursor Positions
                    self.cursPos.x, self.cursPos.y = self.cursor_pos[0], self.cursor_pos[1]
                    self.cursorPos_pub.publish(self.cursPos)

            except KeyboardInterrupt:
                print('[Cursor_node] Stopped interface')

            self.rate.sleep()

    @property
    def p_home(self):
        if self.home_loc == 'center':
            return np.array([self.d1//2,self.d2//2]).astype(int)

if __name__ == "__main__":

    params = load_yaml(f"{bomicontrol_path}/cfg/cursor.yaml")
    print("[Cursor_node] Params:", params)

    cursor_gui = Cursor_interface(params)
    # time.sleep(15)
    cursor_gui.run()