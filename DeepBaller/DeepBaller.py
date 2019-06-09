import tensorflow as tf
import tflearn
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

FRAME_KEEP = 10
FEATURES_LENGTH = 45
OUTPUT_MOVE = 9
OUTPUT_ACT = 5
LR = 1e-3

def DeepBallerMovement():
    tflearn.init_graph(soft_placement = True)
    with tf.device('/gpu:0'):
        network = tflearn.input_data(shape=[None, FRAME_KEEP, FEATURES_LENGTH], name='input')
        network = tflearn.gru(network, 256, return_seq=True, name='DBMove_layer1')
        network = tflearn.dropout(network, 0.6, name='DBMove_layer2')
        network = tflearn.gru(network, 256, return_seq=False, name='DBMove_layer3')
        network = tflearn.dropout(network, 0.6, name='DBMove_layer4')
        network = tflearn.fully_connected(network, OUTPUT_MOVE, activation='softmax', name='DBMove_layer5')
        network = tflearn.regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='DBMove_layer6')
        return tflearn.DNN(network, max_checkpoints=5, tensorboard_verbose=0, checkpoint_path = 'movement_model/movement_model.tfl.ckpt')
    
def DeepBallerAction():
    tflearn.init_graph(soft_placement = True)
    with tf.device('/gpu:0'):
        network = tflearn.input_data(shape=[None, FRAME_KEEP, FEATURES_LENGTH], name='input')
        network = tflearn.gru(network, 256, return_seq=True, name='DBAct_layer1')
        network = tflearn.dropout(network, 0.6, name='DBAct_layer2')
        network = tflearn.gru(network, 256, return_seq=False, name='DBAct_layer3')
        network = tflearn.dropout(network, 0.6, name='DBAct_layer4')
        network = tflearn.fully_connected(network, OUTPUT_ACT, activation='softmax', name='DBAct_layer5')
        network = tflearn.regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='DBAct_layer6')
        return tflearn.DNN(network, max_checkpoints=5, tensorboard_verbose=0, checkpoint_path = 'action_model/action_model.tfl.ckpt')
    
def DeepBaller():
    tflearn.init_graph(soft_placement = True)
    with tf.device('/gpu:0'):
        network = tflearn.input_data(shape=[None, FRAME_KEEP, FEATURES_LENGTH], name='input')
        network = tflearn.gru(network, 256, return_seq=True, name='DBFull_layer1')
        network = tflearn.dropout(network, 0.6, name='DBFull_layer2')
        network = tflearn.gru(network, 256, return_seq=True, name='DBFull_layer3')
        network = tflearn.dropout(network, 0.6, name='DBFull_layer4')
        movement_network = tflearn.gru(network, 256, return_seq=False, name='DBMove_layer1')
        movement_network = tflearn.dropout(movement_network, 0.6, name='DBMove_layer2')
        movement_network = tflearn.fully_connected(movement_network, OUTPUT_MOVE, activation='softmax', name='DBMove_layer3')
        movement_network = tflearn.regression(movement_network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='DBMove_layer4')
        action_network = tflearn.gru(network, 256, return_seq=False, name='DBAct_layer1')
        action_network = tflearn.dropout(action_network, 0.6, name='DBAct_layer2')
        action_network = tflearn.fully_connected(action_network, OUTPUT_ACT, activation='softmax', name='DBAct_layer3')
        action_network = tflearn.regression(action_network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR, name='DBAct_layer4')
        network = tflearn.merge([movement_network, action_network], mode='concat', name='DBFull_layer5')
        return tflearn.DNN(network, max_checkpoints=5, tensorboard_verbose=0, checkpoint_path = 'full_model/full_model.tfl.ckpt')
    
def DataToLSTM(inputs, output_move, output_act):
    trainX = np.empty([FRAME_KEEP, FEATURES_LENGTH], dtype = float)
    trainY_move = np.empty([OUTPUT_MOVE,], dtype = float)
    trainY_act = np.empty([OUTPUT_ACT,], dtype = float)
    
    for i in range(0, np.shape(inputs)[0] - FRAME_KEEP):
        window = inputs[i:i + FRAME_KEEP]
        
        sampleX = []
        for row in window:
            sampleX.append(row)
        sampleY_move = output_move[i + FRAME_KEEP]
        sampleY_act = output_act[i + FRAME_KEEP]
        
        sampleX = np.array(sampleX)
        sampleX = sampleX.reshape((FRAME_KEEP, FEATURES_LENGTH))
        sampleY_move = np.array(sampleY_move)
        sampleY_act = np.array(sampleY_act)
        sampleY_move = sampleY_move.reshape((OUTPUT_MOVE,))
        sampleY_act = sampleY_act.reshape(OUTPUT_ACT,)
        
        trainX = np.dstack((trainX, sampleX))
        trainY_move = np.dstack((trainY_move, sampleY_move))
        trainY_act = np.dstack((trainY_act, sampleY_act))
        
        

    trainX = np.swapaxes(trainX, 0, 2)
    trainY_move = np.swapaxes(trainY_move, 0, 2)
    trainY_act = np.swapaxes(trainY_act, 0, 2)
    trainX = np.swapaxes(trainX, 1, 2)         
    
    trainY_move = trainY_move.reshape(-1, OUTPUT_MOVE)
    trainY_act = trainY_act.reshape(-1, OUTPUT_ACT)
    
    return trainX, trainY_move, trainY_act
        