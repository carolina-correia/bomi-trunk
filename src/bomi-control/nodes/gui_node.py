#!/usr/bin/env python3
import os
" @author: Carolina Correia, @email:cgprcorreia@gmail.com "

"""
GUI with 3 modalities: free cursor motion, training and path-following
--> 1. Free cursor: cursor motion control with trunk alone
--> 2. In Training, a timer is activated to record trunk trajectories with same length and initial position
--> 3. In Path-following, it receives a trajectory and displays is gradually for user to follow 
"""

import numpy as np
import cv2, time, pyautogui, sys
import rospy
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Int32, Float64MultiArray

# define paths
bomi_ws_path = os.environ.get('BOMI_WS', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) + '/')
bomicontrol_path = bomi_ws_path +"src/bomi-control"
sys.path.append(bomicontrol_path)

from cfg.tools import get_subj_day_folder, load_yaml
from interface.displays import *
from interface.tools import *
DARK_RED = (0, 0, 125)
# class for visualizing cursor and trajectories
class Graphical_Interface:
    def __init__(self, params, trajectories=None, time_ref=None)-> None:

        rospy.init_node('gui_node', anonymous=True)

        # Publish and subcribe to topics
        self.cursorPos_sub = rospy.Subscriber('/cursor_pos',Pose2D, self.cursorPos_listener)
        self.calibDone_sub = rospy.Subscriber('/calibrated',Int32, self.calib_listener)
        self.recalibIMU_pub= rospy.Publisher('/recalibrate_IMU',Int32, queue_size=1)
        self.trial_pub     = rospy.Publisher('/trial_ID',Int32, queue_size=10)
        self.trajID_pub    = rospy.Publisher('/traj_ID',Int32, queue_size=10)

        self.ref_traj_pub = rospy.Publisher('/ref_trajectories',Float64MultiArray, queue_size=100)
        self.ref_time_pub = rospy.Publisher('/ref_time',Float64MultiArray, queue_size=100)
        self.targets_list_pub = rospy.Publisher('/targets_list',Float64MultiArray, queue_size=100)

        self.cursorPosReceived = False 
        self.calib_done  = False
        self.recalib_msg = Int32()
        self.trial_id_msg = Int32()
        self.traj_id_msg  = Int32()
        self.ref_traj_msg, self.ref_time_msg, self.targets_list_msg = Float64MultiArray(), Float64MultiArray(), Float64MultiArray()

        # Interface
        self.d1, self.d2 = pyautogui.size() # w, h     
        self.r_target = 12   
        self.r_cursor = params['interface']['r_cursor']
        self.r_circle = params['interface']['r_circle']
        self.home_loc = params['interface']['home_loc']
        self.num_reps = params['interface']['n_repetitions'] #for each trajectory
        self.freq = params['interface']['freq']
        self.rate = rospy.Rate(self.freq)
        self.img = self.background

        # Task
        self.trial_duration = params['interface']['trial_duration']
        self.trial_time = 0
        self.time_inside_target = 0.5
        self.time_inside_home = 4
        self.time_left_at_home = 3
        self.trial_id = 1
        self.num_trials_completed = 0
        self.recalibrate = params['interface']['imu_recalibration']

        # Trajectory task
        self.trajectories = trajectories

        if self.trajectories is not None:
            self.time_vector = time_ref
            self.traj_duration = time_ref[-1]
            self.nb_traj = len(trajectories)
            self.color_traj = STEEL_BLUE
            self.line_thickness = params['interface']['traj_thickness']

            # generate random targets_list
            self.targets_list = np.concatenate([np.random.permutation(np.arange(self.nb_traj)) for _ in range(self.num_reps)])
            # self.targets_list = np.tile(np.arange(self.nb_traj),self.num_reps)
            # self.targets_list = np.tile(2,self.num_reps)
            # self.targets_list.sort()
            self.curr_traj_idx = -1
            self.update_trajectory()

    def publish_traj_info(self):
        self.ref_traj_msg.data = np.float64(self.trajectories.reshape(-1).tolist())
        self.ref_time_msg.data = np.float64(self.time_vector.tolist())
        self.targets_list_msg.data = np.float64(self.targets_list.tolist()) 
        self.ref_traj_pub.publish(self.ref_traj_msg)  
        self.ref_time_pub.publish(self.ref_time_msg)
        self.targets_list_pub.publish(self.targets_list_msg)

    def cursorPos_listener(self,msg):
        self.cursor_pos = np.array([msg.x, msg.y]).astype(int)
        self.cursorPosReceived = True

    def calib_listener(self,msg):
        """ After sending recalib request to hmi_imu, waits to see if calib was done"""
        self.calib_done = msg.data

    def update_trajectory(self):
        # update trajectory
        self.img = self.background
        self.curr_traj_idx += 1
        self.curr_target_traj = self.targets_list[self.curr_traj_idx]
        self.curr_traj = self.trajectories[self.curr_target_traj,:,:].astype(int)

    def display_traj_segment(self,idx_start,idx_end):
        segment = self.curr_traj[idx_start:idx_end,:]
        self.img = cv2.polylines(self.img, [segment], False, color=BLUE, thickness=self.line_thickness)

    def display_cursor_track(self,segment):
        self.img = cv2.polylines(self.img, [segment], False, RED, thickness=15)

    def plot_trajectory(self,k):
        self.img = self.background
        self.img = cv2.polylines(self.img, [self.curr_traj], False, color=self.color_traj, thickness=self.line_thickness+10)
        # self.img = cv2.circle(self.img, tuple(self.curr_traj[k+3]), self.r_cursor*1.5, RED, -1)  # new cursor pos

        # cv2.circle(self.img, tuple(self.curr_traj[k]), self.r_target, WHITE, -1)  # White interior
        # self.img = cv2.addWeighted(self.img, 1 - 0.2, self.img, 0.2, 0)
        # self.img = cv2.circle(self.img, tuple(self.curr_traj[k]), self.r_target, DARK_RED, thickness=3)  # Red border
        
        # cursor
        # self.img = cv2.circle(self.img, tuple(self.cursor_pos), self.r_cursor, RED, -1)  # new cursor pos
        # self.img = cv2.circle(self.img, tuple(self.curr_traj[k]), self.r_cursor, RED, -1)  # new cursor pos

    def plot_cursor_trace_on_traj(self):
        """ plot new point position, with same trajectory there """
        self.img = cv2.circle(self.img, tuple(self.cursor_pos), self.r_cursor, RED, -1)  # new cursor pos

    def set_recalibration_screen(self):
        """ plot just home position and cursor pos"""
        self.img = self.background
        self.img = user_info("Do not move")
        # self.img = cv2.circle(self.img, tuple(self.cursor_pos), self.r_cursor, RED, -1)  # new cursor pos

    def plot_home(self):
        overlay = self.background.copy()
        cv2.circle(overlay, self.p_home, radius=self.r_circle, color=BLUE, thickness=-1)
        img = cv2.addWeighted(overlay, 0.2, self.background, 1-0.2, 0, self.background)
        self.img = cv2.circle(img, self.p_home, radius=self.r_circle, color=BLUE, thickness=4)

    def plot_target(self,color,radius):
        # overlay = self.img.copy()
        # cv2.circle(overlay, self.p_target, radius=self.r_circle, color=color, thickness=-1)
        # img = cv2.addWeighted(overlay, alpha, self.img, 1-alpha, 0, self.background)
        self.img = cv2.circle(self.img, self.p_target, radius=radius, color=color, thickness=3)

    def plot_cursor(self):
        self.img = cv2.circle(self.img, tuple(self.cursor_pos), self.r_cursor, RED, -1)  # new cursor pos

    def plot_cursor_and_home(self,home_color=None):
        """ plot new point position, with same trajectory there """
        overlay = self.background.copy()
        c = BLUE if home_color is None else home_color
        cv2.circle(overlay, self.p_home,  radius=self.r_circle, color=c, thickness=-1)
        img = cv2.addWeighted(overlay, 0.2, self.background, 1-0.2, 0, self.background)
        img = cv2.circle(img, self.p_home, radius=self.r_circle, color=c, thickness=4)
        self.img = cv2.circle(img, tuple(self.cursor_pos), self.r_cursor, RED, -1)  # new cursor pos

    def run_path_following(self):
        open_window('GUI','Cursor', self.d1, self.d2)
        set_display('GUI', full_scrn=True)
        reached_home, target_reached, start_task = False, False, False
        k = 0
        trial_label = 0
        t_start = time.time()
        self.r_circle = 15 #smaller

        while not rospy.is_shutdown():
            try:
                if time.time() - t_start < 10:
                    self.publish_traj_info()

                if not self.calib_done:
                    self.set_recalibration_screen()

                if self.calib_done and self.cursorPosReceived:
                    if not reached_home:
                        if self.dist_to_home <= self.r_circle:
                            print("[GUI_node] Cursor inside home. Starting timer")
                            self.t_reached_home = time.time()
                            reached_home = True

                    if reached_home and not start_task:
                        if self.dist_to_home > self.r_circle:
                            reached_home = False
                        if (time.time()-self.t_reached_home) <= self.time_inside_home:
                            self.time_left_at_home = self.time_inside_home - (time.time()-self.t_reached_home)
                        else:
                            print("[GUI_node] Cursor left home. Starting task")
                            self.time_start_task = time.time()
                            self.plot_home()
                            start_task = True

                    if start_task:
                        self.trial_time = time.time() - self.time_start_task
                        trial_label = self.trial_id

                        if k == 0:
                            self.plot_trajectory(k)
                        idx_start = k-2 if k > 2 else 0
                        print(k)
                        if k <= len(self.curr_traj):
                            self.display_traj_segment(idx_start=idx_start,idx_end=k)
                            self.plot_cursor()
                            k += 1
                        if k == len(self.curr_traj):
                            self.plot_target(color=RED,radius=12)

                        self.plot_cursor_trace_on_traj()
                        if not target_reached and k >= len(self.curr_traj)-1:
                            # print(self.dist_to_target)
                            if self.dist_to_target <= self.r_target:
                                print("[GUI_node] Cursor inside target. Starting timer")
                                self.t_reached_target = time.time()
                                self.plot_target(color=RED,radius=12)
                                target_reached = True

                        if target_reached:
                            self.plot_cursor_and_home(home_color=BLUE)
                            print("[GUI_node] Task completed!")
                            # self.set_recalibration_screen()
                            self.update_trajectory()                    
                            self.recalib_msg.data = 1
                            self.calib_done = 0
                            self.num_trials_completed += 1
                            self.trial_id += 1
                            self.time_start_task = 0
                            trial_label = 0
                            reached_home, target_reached, start_task = False, False, False
                            k=0  
                            # time.sleep(2)   
                    else:
                        self.trial_time = 0
                        self.plot_cursor_and_home()

                    # time.sleep(0.1)
                    if reached_home and not start_task:
                        self.img = display_timer(self.img,self.time_left_at_home,self.p_home,'GO')
                    
                # reinitialize cursor position: ask imu interface to recalibrate, waits for calib_done flag
                if self.recalibrate:
                    self.recalibIMU_pub.publish(self.recalib_msg)
                    self.recalib_msg.data = 0

                # Publish trial ID
                self.trial_id_msg.data = trial_label
                self.trial_pub.publish(self.trial_id_msg)

                # Publish trajectory ID
                self.traj_id_msg.data = self.curr_target_traj
                self.trajID_pub.publish(self.traj_id_msg)

                self.img = add_number_of_trials(self.img,self.num_trials_completed,self.d1)
                self.img = add_time_passed(self.img,self.trial_time,self.d1,self.d2,loc='bottom right',color='black')

                cv2.imshow('GUI', self.img)
                if cv2.waitKey(1) == 27:
                    print('[GUI_node] Stopped interface')
                    cv2.destroyAllWindows()
                    return
            except KeyboardInterrupt:
                print('[GUI_node] Stopped interface')
            self.rate.sleep()

    def run_simulation(self):
        """ Check if speed of trajectories is ok"""
        open_window('GUI','Cursor', self.d1, self.d2)
        set_display('GUI', full_scrn=True)
        target_reached, start_task =False, True
        self.time_start_task = time.time()
        k = 0
        c = -50
        while not rospy.is_shutdown():
            try:
                if start_task:
                    if target_reached:
                        print("[GUI_node] Task completed!")
                        self.set_recalibration_screen()
                        self.update_trajectory()                    
                        self.num_trials_completed += 1
                        self.trial_id += 1
                        self.time_start_task = time.time()
                        target_reached =False
                        k=0
                        c=-50

                    self.trial_time = time.time() - self.time_start_task

                    if k == 0:
                        self.plot_trajectory(k)
                    # k+=1
                    idx_start = k-2 if k > 2 else 0
                    if k <= len(self.curr_traj):
                        self.display_traj_segment(idx_start=idx_start,idx_end=k)
                        self.img = cv2.circle(self.img, tuple(self.curr_traj[0,:]), 10, RED, -1)  # new cursor pos
                        k += 1
                    if 3 <= c <= len(self.curr_traj):
                        self.display_cursor_track( self.curr_traj[:c,:])
                    c+= 1

                    if k >= len(self.curr_traj) and c >= len(self.curr_traj):
                        self.plot_target(color=RED,radius=self.r_cursor)
                        target_reached = True

                self.img = add_number_of_trials(self.img,self.num_trials_completed,self.d1)
                self.img = add_time_passed(self.img,self.trial_time,self.d1,self.d2,loc='bottom right',color='black')

                cv2.imshow('GUI', self.img)
                if cv2.waitKey(1) == 27:
                    print('[GUI_node] Stopped interface')
                    cv2.destroyAllWindows()
                    return
                
                if target_reached:
                    time.sleep(2)
            except KeyboardInterrupt:
                print('[GUI_node] Stopped interface')
            self.rate.sleep()


    def run_training(self):
        open_window('GUI','Cursor', self.d1, self.d2)
        set_display('GUI', full_scrn=True)
        reached_home, start_task = False, False
        trial_label = 0
        self.r_circle = 15 #smaller
        while not rospy.is_shutdown():
            try:
                if not self.calib_done:
                    self.set_recalibration_screen()

                if self.calib_done and self.cursorPosReceived:
                    if not reached_home:
                        if self.dist_to_home <= self.r_circle:
                            print("[GUI_node] Cursor inside home. Starting timer")
                            self.t_reached_home = time.time()
                            reached_home = True

                    if reached_home and not start_task:
                        if self.dist_to_home > self.r_circle:
                            reached_home = False
                        if (time.time()-self.t_reached_home) <= self.time_inside_home:
                            self.time_left_at_home = self.time_inside_home - (time.time()-self.t_reached_home)
                        else:
                            self.time_start_task = time.time()
                            self.plot_home()
                            start_task = True

                    if start_task:
                        self.trial_time = time.time() - self.time_start_task
                        time_left = np.round(self.trial_duration - self.trial_time)
                        trial_label = self.trial_id
                        self.img = display_training_time(self.background,time_left,self.d1,self.d2)
                        # self.plot_cursor_trace_on_traj()

                        if self.trial_time >= self.trial_duration:
                            self.img = self.background
                            self.recalib_msg.data = 1
                            self.calib_done = 0                            
                            self.num_trials_completed += 1
                            self.trial_id += 1
                            self.time_start_task = 0
                            trial_label = 0
                            reached_home, start_task = False, False
                            print("[GUI_node] Trial completed!")
                            # time.sleep(2)
                    else:
                        self.trial_time = 0
                        self.plot_cursor_and_home()

                    if reached_home and not start_task:
                        self.img = display_timer(self.img,self.time_left_at_home,self.p_home,'GO')
                    
                # reinitialize cursor position: ask imu interface to recalibrate, waits for calib_done flag
                if self.recalibrate:
                    self.recalibIMU_pub.publish(self.recalib_msg)
                    self.recalib_msg.data = 0

                # Publish trial ID
                self.trial_id_msg.data = trial_label
                self.trial_pub.publish(self.trial_id_msg)

                self.img = add_number_of_trials(self.img,self.num_trials_completed,self.d1)

                cv2.imshow('GUI', self.img)
                if cv2.waitKey(1) == 27:
                    print('[GUI_node] Stopped interface')
                    cv2.destroyAllWindows()
                    return
            except KeyboardInterrupt:
                print('[GUI_node] Stopped interface')
            self.rate.sleep()

    def run_free_cursor(self):
        open_window('Cursor control','Cursor', self.d1, self.d2)
        set_display('Cursor control', full_scrn=True)
        while not rospy.is_shutdown():
            try:
                if self.cursorPosReceived:
                    self.plot_cursor()
                    cv2.imshow('Cursor control', self.img)
                    if cv2.waitKey(1) == 27: 
                        print('[GUI_node] Stopped interface ')
                        cv2.destroyAllWindows()
                        return
            except KeyboardInterrupt:
                print('[GUI_node] Stopped interface')
            self.rate.sleep()

    @property
    def offset(self):
        if self.home_loc == "center":
            return np.array([self.d1//2, self.d2//2])
    @property
    def p_home(self):
        # Get it displaced
        if self.trajectories is not None:
            return get_home_center(self.curr_traj,self.r_circle, m=20)
        else:
            return (self.d1//2, self.d2//2)

    @property
    def p_target(self):
        if self.trajectories is not None:
            # vec = self.curr_traj[-1] - self.curr_traj[-5]
            # p_target = self.curr_traj[-1] - (self.r_circle+1)*(vec/(1e-5+np.linalg.norm(vec)))
            # return tuple(p_target.astype(int))
            return self.curr_traj[-1]
        else:
            return

    @property
    def dist_to_home(self):
        #distance between cursor pos center and current target
        return np.linalg.norm(self.cursor_pos-self.p_home)
    @property
    def dist_to_target(self):
        return np.linalg.norm(self.cursor_pos-self.p_target)
    @property
    def background(self):
        return np.ones((self.d2,self.d1,3),np.uint8)*255 # white

if __name__ == "__main__":
    print("Starting GUI...")

    params = load_yaml(f"{bomicontrol_path}/cfg/params.yaml")
    subj_day_folder = get_subj_day_folder(params['load']['subj'],params['load']['date'])

    if params['interface']['task'] == 'free':
        gui = Graphical_Interface(params,trajectories=None)
        gui.run_free_cursor()

    if params['interface']['task'] == 'training':
        gui = Graphical_Interface(params,trajectories=None)
        gui.run_training()

    if params['interface']['task'] == 'path_following':
        # receives trajectories already scaled to screen
        t_ref        = np.load(f"{subj_day_folder}training/time.npy")
        trajectories = np.load(f"{subj_day_folder}interface/traj_screen_pred.npy")
        if len(params['interface']['traj_id'])>0:
            trajectories = trajectories[params['interface']['traj_id'],:,:]
        t_ref = t_ref[-trajectories.shape[1]:] - t_ref[0]
        traj_resamp, time_resamp = resample_traj(t_ref,trajectories,params['interface']['freq_trajectory'])
        gui = Graphical_Interface(params,traj_resamp,time_resamp)
        gui.run_path_following()

    if params['interface']['task'] == 'sim':
        t_ref        = np.load(f"{subj_day_folder}training/time.npy")
        trajectories = np.load(f"{subj_day_folder}interface/traj_screen_pred.npy")
        if len(params['interface']['traj_id'])>0:
            trajectories = trajectories[params['interface']['traj_id'],:,:]
        t_ref = t_ref[-trajectories.shape[1]:] - t_ref[0]
        traj_resamp, time_resamp = resample_traj(t_ref,trajectories,params['interface']['freq_trajectory'])
        gui = Graphical_Interface(params,traj_resamp,time_resamp)
        gui.run_simulation()
