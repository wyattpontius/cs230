import numpy as np
import time
import cv2
from DeepBaller import DeepBaller
from pydarknet import Detector, Image
from pymouse import PyMouse
from pykeyboard import PyKeyboard

FRAME_KEEP = 10
FEATURES_LENGTH = 45
OUTPUT_MOVE = 9
OUTPUT_ACT = 5
LR = 1e-3
DUMMY = -3

net = Detector(bytes("/home/wyatt/darknet/cfg/yolov3-2K.cfg", encoding="utf-8"), bytes("/home/wyatt/darknet/weights/yolov3-2K_6000.weights", encoding="utf-8"), 0, bytes("/home/wyatt/darknet/cfg/2K.data",encoding="utf-8"))

cap = cv2.VideoCapture(0)

model = DeepBaller()

model.load('/home/wyatt/full_model/zero_loss/full_model_all_data.tfl')

keyboard = PyKeyboard()

fps = 10
secs_per_frame = 1.0/fps
prev_time = time.time()

input_X = np.full((1, FRAME_KEEP, FEATURES_LENGTH), DUMMY)

def take_movement(keys_to_move):
    all_keys = ['W', 'A', 'S', 'D']
    for key in all_keys:
        if key in keys_to_move:
            keyboard.press_key(key)
        else:
            keyboard.release_key(key)
            
def take_action(action_keys):
    # the following array maps to ['A', 'X', 'Y', 'Left bumper'] on the Xbox controller
    all_keys = ['space', 'R', '1', 'Q']
    for key in all_keys:
        if key in action_keys:
            keyboard.press_key(key)
        else:
            keyboard.release_key(key)

while(True):
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
        
        if len(final) != 45:
            continue
        else:
            final = np.asarray(final)
            final = np.reshape(final, (1, 45))
            
            if np.any(input_X == DUMMY):
                i = 0
                while np.any(input_X[0][i] != DUMMY):
                    i += 1
                input_X[i] = final
            else:
                for i in range(0, FRAME_KEEP):
                    if i == FRAME_KEEP - 1:
                        input_X[0][i] = final
                    else:
                        input_X[0][i] = input_X[0][i + 1]
                    
            if np.any(input_X[0][9] != DUMMY):
                prediction_move = model.predict(input_X)[0][0:9]
                prediction_act = model.predict(input_X)[0][9:14]
                prediction_move = [7, 4, 7, 4, 5, 5, 8, 3, 1] * prediction_move
                prediction_act = [20, 25, 10, 3, 1] * prediction_act
                move_choice = np.argmax(prediction_move)
                act_choice = np.argmax(prediction_act)
                
                if move_choice == 0:
                    take_movement(['W', 'D'])
                elif move_choice == 1:
                    take_movement(['S', 'D'])
                elif move_choice == 2:
                    take_movement(['W', 'A'])
                elif move_choice == 3:
                    take_movement(['S', 'A'])
                elif move_choice == 4:
                    take_movement(['D'])
                elif move_choice == 5:
                    take_movement(['A'])
                elif move_choice == 6:
                    take_movement(['W'])
                elif move_choice == 7:
                    take_movement(['S'])
                elif move_choice == 8:
                    take_movement([])
                
                if act_choice == 0:
                    take_action(['space'])
                elif act_choice == 1:
                    take_action(['R'])
                elif act_choice == 2:
                    take_action(['1'])
                elif act_choice == 3:
                    take_action(['Q'])
                elif act_choice == 4:
                    take_action([])
            
            print(prediction_move)
            print(move_choice)
            print(prediction_act)
            print(act_choice)
            
            
        while start_time + secs_per_frame > time.time():
            pass
