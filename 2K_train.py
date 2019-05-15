import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from collections import deque
from models import inception_v3 as googlenet
from models import otherception3
from random import shuffle
from tensorflow import convert_to_tensor, int32

# number of testing data files
FILE_I_END = 8

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30

MODEL_NAME = '2k_initial_test'
PREV_MODEL = ''

LOAD_MODEL = False

DEVICE_NAME = 'GPU'
DEVICE_NUM = '0'

model = googlenet(WIDTH, HEIGHT, 3, LR, output=6, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
    

# iterates through the training files


for e in range(EPOCHS):
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        
        try:
            file_name = 'C:/Users/pablo/Documents/Python Scripts/2K_data/training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name, allow_pickle = True)
            X = []
            Y = []
            for entry in train_data:
                X.append(entry[0].reshape(WIDTH,HEIGHT,3))
                Y.append(np.asarray(entry[1], dtype=float))
            
           # train_data = np.asarray(train_data, dtype = float)
            
           # print (train_data)
           # print('training_data-{}.npy'.format(i),len(train_data))

##            # [   [    [FRAMES], CHOICE   ]    ] 
##            train_data = []
##            current_frames = deque(maxlen=HM_FRAMES)
##            
##            for ds in data:
##                screen, choice = ds
##                gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
##
##
##                current_frames.append(gray_screen)
##                if len(current_frames) == HM_FRAMES:
##                    train_data.append([list(current_frames),choice])


            # #
            # always validating unique data: 
            #shuffle(train_data)
            train_x = X[:-50]
            train_y = Y[:-50]
            
            train_x = np.array(train_x, dtype = float)
            train_y = np.array(train_y, dtype = float)
            
            test_x = X[-50:]
            test_y = Y[-50:]
            
            test_x = np.array(test_x, dtype = float)
            test_y = np.array(test_y, dtype = float)

           # X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
            #Y = np.array([i[1] for i in train])
            # print (str(Y))

            #test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
            #test_y = np.array([i[1] for i in test])
            model.fit({'input': train_x}, {'targets': train_y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
                snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)


            if count%10 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                    
        except Exception as e:
            print(str(e))
            
    