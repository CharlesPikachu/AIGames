'''
Function:
    Define the deep q network
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten


'''define the network'''
def buildNetwork(imagesize, in_channels, num_actions, **kwargs):
    dqn_model = Sequential([Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(*imagesize, in_channels)),
                            Activation('relu'),
                            Conv2D(64, (4, 4), padding='same', strides=(2, 2)),
                            Activation('relu'),
                            Conv2D(64, (3, 3), padding='same', strides=(1, 1)),
                            Activation('relu'),
                            Flatten(),
                            Dense(512),
                            Activation('relu'),
                            Dense(num_actions)])
    return dqn_model