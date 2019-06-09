from DeepBaller import DeepBaller
import numpy as np

model = DeepBaller()

MODEL_NAME = '2k_full_model'
FRAME_KEEP = 10
lr = 1e-3
OUTPUT_MOVE = 9
OUTPUT_ACT = 5
EPOCHS = 99

X = np.load('/home/wyatt/Documents/2KData_Full/cleaned_lstm_data-X.npy')
Ymove = np.load('/home/wyatt/Documents/2KData_Full/cleaned_lstm_data-Ymove.npy')
Yact = np.load('/home/wyatt/Documents/2KData_Full/cleaned_lstm_data-Yact.npy')

X = X[1:100001]
Ymove = Ymove[1:100001]
Yact = Yact[1:100001]

for e in range(EPOCHS): 

    train_X = np.array(X[0+(1000*e):1000+(1000*e)], dtype = float)
    train_Y_move = np.array(Ymove[0+(1000*e):1000+(1000*e)], dtype = float)
    train_Y_act = np.array(Yact[0+(1000*e):1000+(1000*e)], dtype = float)
    
    test_X = X[-200:]
    test_Y_move = Ymove[-200:]
    test_Y_act = Yact[-200:]
    
    test_X = np.array(test_X, dtype = float)
    test_Y_move = np.array(test_Y_move, dtype = float)
    test_Y_act = np.array(test_Y_act, dtype = float)   
               
    model.fit(train_X, [train_Y_move, train_Y_act], n_epoch=5, 
               validation_set=(test_X, [test_Y_move, test_Y_act]), 
               show_metric=True, batch_size = 200)
    model.save('full_model/full/full_model_all_data.tfl')