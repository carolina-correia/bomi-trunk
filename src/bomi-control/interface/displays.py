#!/usr/bin/env python3
" @author: Carolina Correia, @email:cgprcorreia@gmail.com "
"""
This script has modules for generating trajectories and testing it
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2, time, pyautogui
import sys

if sys.platform != 'darwin':
    import rospy
    from geometry_msgs.msg import Pose2D

# Build screen config
WHITE = (255, 255, 255)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
BLACK = (0, 0, 0)
LIGHT_BLUE = (153, 153, 0)
STEEL_BLUE = (230,216,173)
GREEN = (96, 178, 14)


def open_window(window_name, title, width=None, height=None):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(window_name, title)
    if width and height: cv2.resizeWindow(window_name, width, height)

def set_display(window_name, full_scrn):
    """Set disply window to either full screen or normal."""
    if full_scrn: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else: cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def user_info(text):
    d1, d2 = pyautogui.size()
    black_screen = np.ones((d2,d1,3),np.uint8)*255  # dark background (*255 for white)
    font = cv2.FONT_HERSHEY_DUPLEX
    textsize, fontscale = cv2.getTextSize(text,font,1,2)[0], 1
    x, y = int((d1-textsize[0])/2), int((d2-textsize[1])/2)
    screen = cv2.putText(black_screen,text,(x,y),font,fontscale,BLACK,2)
    return screen

def add_number_of_trials(img,nb_trials_completed,d1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = WHITE  # black color
    font_thickness = 2    

    text = str(nb_trials_completed)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size
    x, y = d1 - text_width, 20
    alpha = 0.3  # Adjust the alpha value for transparency

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y+5), (x + text_width+20, y - text_height - 50), BLACK, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    img = cv2.putText(img, text, (x, y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return img

def add_time_passed(img,time_passed,d1,d2,loc,color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = RED if color == 'red' else BLACK
    time_passed_sec = int(np.round(time_passed))
    text = str(time_passed_sec)
    if loc == 'bottom right':
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        x, y = d1 - text_width, d2 - text_height
        start_point, end_point = (x, y+5), (x + text_width+20, y - text_height - 20)
        text_x, text_y = x,y
    if loc == 'top left':
        text = str(time_passed_sec) if time_passed_sec != 0 else 'STOP'
        font_scale, font_thickness = 2, 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_width, text_height = text_size
        x, y = 5, 30 + text_height
        start_point, end_point = (x, y), (x + text_width + 20, y - text_height - 20)
        text_x, text_y = start_point[0] + 15, start_point[1] - 15
    cv2.rectangle(img, start_point, end_point, WHITE, -1)
    if color != 'white':
        img = cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
    return img


def display_timer(img, time_passed, pos, command):
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_passed_sec = int(time_passed)
    if time_passed_sec == 0:
        text = command
    else:
        text = str(time_passed_sec)
    text_size = cv2.getTextSize(text, font, 3,5)[0]
    text_x = pos[0] - text_size[0]
    text_y = pos[1] + text_size[1]
    img = cv2.putText(img, text, (text_x, text_y),font,3,BLACK,5, cv2.LINE_AA)
    return img


def display_training_time(img, time_passed, d1, d2):
    img = img.copy()
    color = GREEN
    radius = 400
    font_size = 4
    cv2.circle(img, (d1//2, d2//2), radius, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_passed_sec = int(time_passed)
    if time_passed_sec == 0:
        text = "STOP"
    else:
        text = str(time_passed_sec)
    text_size = cv2.getTextSize(text, font, 3,5)[0]
    text_x, text_y = (d1 - text_size[0]) // 2, (d2 + text_size[1]) // 2
    img = cv2.putText(img, text, (text_x, text_y),font,font_size,WHITE,5, cv2.LINE_AA)
    return img