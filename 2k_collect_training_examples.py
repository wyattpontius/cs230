# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:19:05 2019

@author: pablo
"""

import math
from inputs import get_gamepad
import threading
import time
from PIL import Image
from grabscreen import grab_screen
import cv2
import numpy as np
import os

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self):
        joystick_x = self.LeftJoyStickX
        joystick_y = self.LeftJoyStickY
        a = self.A
        x = self.X
        lb = self.LeftBumper
        rt = self.RightTrigger
        
        if a == 1:
            return [0, 0, 1, 0, 0, 0, 0, 0]
        if x == 1:
            return [0, 0, 0, 1, 0, 0, 0, 0]
        if joystick_x >= 0.2:
            return [1, 0, 0, 0, 0, 0, 0, 0]
        if joystick_y >= 0.2:
            return [0, 1, 0, 0, 0, 0, 0, 0]
        if joystick_x <= -0.2:
            return [0, 0, 0, 0, 0, 0, 1, 0]
        if joystick_x <= -0.2:
            return [0, 0, 0, 0, 0, 0, 0, 1]
        
        return [joystick_x, joystick_y, a, x, lb, rt, joystick_x, joystick_y]
# =============================================================================
#         left_x = self.LeftJoystickX
#         left_y = self.LeftJoystickY
#         a = self.A
#         x = self.X # b=1, x=2
#         rt = self.RightTrigger
#         lb = self.LeftBumper
#         return [left_x, left_y, a, x, rt, lb]
# =============================================================================


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state
 


xbox_controller = XboxController()
                   
fps = 60
secs_per_frame = 1.0/fps
prev_time = time.time()

starting_value = 1

while True:
    file_name = 'C:/Users/pablo/Documents/Python Scripts/2K_data/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File {} exists, moving along'.format(starting_value))
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break

file_name = 'C:/Users/pablo/Documents/Python Scripts/2K_data/training_data-{}.npy'.format(starting_value)

training_data = []

while True:
    start_time = time.time()
    # get screen shot, flatten into nd array
    
    # TODO: make sure that the region is capturing the output of the darknet demo (with bounding boxes)
    # right now this captures ~ bottom right quadrant of the screen
    screen = grab_screen(region= (1920, 50, 3840, 1070))
    last_time = time.time()
    
    # resize to something a bit more acceptable for a CNN
    # need to resize so the processing goes by faster
    screen = cv2.resize(screen, (480,270))
    
    # following two lines are for debugging:
    # they save the screen grabs as PNGs so we can check window size/placement etc
    #im = Image.fromarray(screen)
    #im.save('C:/Users/pablo/Documents/Python Scripts/screenshots/screenshot{}.png'.format(i))
    
   # print('loop took {} seconds'.format(time.time()-last_time))


    # get controller input, we need to convert this to multi-hot arrays to pass as output
    curr_buttons = xbox_controller.read()
    
    training_data.append([screen, curr_buttons])
    # append to results
    if len(training_data) == 1024:
        np.save(file_name,training_data)
        print('SAVED')
        training_data = []
        
        file_name = 'C:/Users/pablo/Documents/Python Scripts/2K_data/training_data-{}.npy'.format(starting_value)   
        starting_value += 1
    
    # wait for next frame
    while start_time + secs_per_frame > time.time():
        pass
    