import numpy as np
from grabscreen import grab_screen
import cv2
import time
from models import inception_v3 as googlenet
from directkeys import PressKey,ReleaseKey, W, A, S, D, SPACE, R, S, E
import pyautogui

keysMapping = {0:'d', 1:'w', 2:'space', 3:'r', 4:'e', 5:'w', 6:'a', 7:'s'}


GAME_WIDTH = 1920
GAME_HEIGHT = 1080

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 10

MODEL_NAME = 'models/2k_initial_test'
model = googlenet(WIDTH, HEIGHT, 3, LR, output=6, model_name = MODEL_NAME)

model.load(MODEL_NAME)

print('We have loaded a previous model!')

fps = 1
secs_per_frame = 1.0/fps

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region= (1920, 50, 3840, 1070))


    count = 0
    prev_key = None
    while(True):
        
        if not paused:
            count += 1
            
            if prev_key:
                pyautogui.keyUp(prev_key)
            
            if count % 5 == 0:
                count = 0
                pyautogui.keyDown('r')
                prev_key = 'r'
                time.sleep(0.5)
                continue
            
            
                
            
            start_time = time.time()              
            screen = grab_screen(region= (1920, 50, 3840, 1070))
            
            last_time = time.time()
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,3)])[0]
            prediction = np.array(prediction) 

            mode_choice = np.argmax(prediction)
# =============================================================================
#             for k, v in keysMapping.items()                                                           :
#                 if k!= mode_choice:
#                     ReleaseKey (v)
#             PressKey (keysMapping[mode_choice])
# =============================================================================
            
            pyautogui.keyDown(keysMapping[mode_choice])
            prev_key = keysMapping[mode_choice]
            # should be between 0-5
            print ('mode_choice: {}'.format(mode_choice))

            print('loop took {} seconds. Choice: {}'.format( round(time.time()-last_time, 3) , mode_choice))
            
            while start_time + secs_per_frame > time.time():
                pass


main()       