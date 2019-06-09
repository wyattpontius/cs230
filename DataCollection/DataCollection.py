from pydarknet import Detector, Image
from inputs import get_gamepad
import cv2
import math
import threading
import time
import os
import numpy as np

net = Detector(bytes("/home/wyatt/darknet/cfg/yolov3-2K.cfg", encoding="utf-8"), bytes("/home/wyatt/darknet/weights/YOLOv3-tiny-2K.weights", encoding="utf-8"), 0, bytes("/home/wyatt/darknet/cfg/2K.data",encoding="utf-8"))

cap = cv2.VideoCapture(0)

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


    def read_movement(self):
        joystick_x = self.LeftJoystickX
        joystick_y = self.LeftJoystickY
        
        if joystick_x >= 0.2 and joystick_y >= 0.2:
            return [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif joystick_x >= 0.2 and joystick_y <= -0.2:
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif joystick_x <= -0.2 and joystick_y >= 0.2:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif joystick_x <= -0.2 and joystick_y <= -0.2:
            return [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif joystick_x >= 0.2:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif joystick_x <= -0.2:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif joystick_y >= 0.2:
            return [0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif joystick_y <= -0.2:
            return [0, 0, 0, 0, 0, 0, 0, 1, 0]
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    def read_action(self):
        a = self.A
        x = self.X
        y = self.Y
        lb = self.LeftBumper
        
        if a == 1:
            return [1, 0, 0, 0, 0]
        elif x == 1:
            return [0, 1, 0, 0, 0]
        elif y == 1:
            return [0, 0, 1, 0, 0]
        elif lb == 1:
            return [0, 0, 0, 1, 0]
        else:
            return [0, 0, 0, 0, 1]
        
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

starting_value = 1

while True:
    file_name = '/home/wyatt/Documents/2KData/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File {} exists, moving along'.format(starting_value))
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        
        break
    
file_name = '/home/wyatt/Documents/2KData/training_data-{}.npy'.format(starting_value)

fps = 10
secs_per_frame = 1.0/fps
prev_time = time.time()

coordinates = []
movements = []
actions = []


while True:
    start_time = time.time()
    r, frame = cap.read()
    if r:
        
        dark_frame = Image(frame)
        results = net.detect(dark_frame)
        
        final = []
        class_array = []
        
        for i in range(0, len(results)):
           string = results[i][0].decode('utf-8')
           class_array.append(string)
        
        if 'basketball\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'basketball\r']
            for j in indices:
                final += [0, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 0:
            final += [0, -1, -1, -1, -1]
        if 'player\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'player\r']
            for j in indices:
                final += [1, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 5:
            final += [1, -1, -1, -1, -1]
        if 'teammate\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'teammate\r']
            for j in indices:
                final += [2, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 10:
            final += [2, -1, -1, -1, -1, 2, -1, -1, -1, -1]
        elif len(final) == 15:
            final += [2, -1, -1, -1, -1]
        if 'opponent\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'opponent\r']
            for j in indices:
                final += [3, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 20:
            final += [3, -1, -1, -1, -1, 3, -1, -1, -1, -1, 3, -1, -1, -1, -1]
        elif len(final) == 25:
            final += [3, -1, -1, -1, -1, 3, -1, -1, -1, -1]
        elif len(final) == 30:
            final += [3, -1, -1, -1, -1]
        if 'hoop\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'hoop\r']
            for j in indices:
                final += [4, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 35:
            final += [4, -1, -1, -1, -1]
        if 'meter\r' in class_array:
            indices = [i for i, x in enumerate(class_array) if x == 'meter\r']
            for j in indices:
                final += [5, results[j][2][0], results[j][2][1], results[j][2][2], results[j][2][3]]
        if len(final) == 40:
            final += [5, -1, -1, -1, -1]
        
        
        
        curr_buttons_movement = xbox_controller.read_movement()
        curr_buttons_action = xbox_controller.read_action()
        
        coordinates.append(final)
        
        movements.append(curr_buttons_movement)
        actions.append(curr_buttons_action)
        
        print(time.time())
        print(final)
        print(curr_buttons_movement)
        print(curr_buttons_action)
        
        if len(movements) == 512:
            np.save(file_name, [coordinates, movements, actions])
            print('SAVED')
            coordinates = []
            movements = []
            actions = []
        
            file_name = '/home/wyatt/Documents/2KData/training_data-{}.npy'.format(starting_value)
        
            starting_value += 1
        
        while start_time + secs_per_frame > time.time():
            pass
    
# =============================================================================
# for cat, score, bounds in results:
#     x, y, w, h = bounds
#     cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), thickness=2)
#     cv2.putText(img,str(cat.decode("utf-8")),(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
# =============================================================================



# =============================================================================
# cv2.imshow("output", img)
# cv2.waitKey(0)
# =============================================================================
