'''
Function:
    define the dqn agent
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import pickle
import random
import skimage
import numpy as np
import skimage.color
import skimage.exposure
import skimage.transform
from collections import deque
from keras.optimizers import Adam
from modules.DQNAgent.dqn import *


'''the dqn agent'''
class DQNAgent():
    def __init__(self, mode, backuppath, **kwargs):
        self.mode = mode
        self.backuppath = backuppath
        # define the necessary variables
        self.num_actions = 2
        self.num_input_frames = 4
        self.discount_factor = 0.99
        self.num_observes = 3200
        self.num_explores = 3e6
        self.epsilon = 0.1
        self.init_epsilon = 0.1
        self.final_epsilon = 1e-4
        self.replay_memory_size = 5e4
        self.imagesize = (80, 80)
        self.save_interval = 5000
        self.num_iters = 0
        self.replay_memory_record = deque()
        self.max_score = 0
        self.input_image = None
        self.batch_size = 32
        # define the dqn network
        self.dqn_model = buildNetwork(imagesize=self.imagesize, in_channels=self.num_input_frames, num_actions=self.num_actions)
        # define the optimizer and loss function
        self.optimizer = Adam(lr=1e-4)
        self.dqn_model.compile(loss='mse', optimizer=self.optimizer)
    '''get the next action'''
    def nextAction(self, reward):
        # some necessary update
        if self.epsilon > self.final_epsilon and self.num_iters > self.num_observes:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.num_explores
        self.num_iters += 1
        # make decision
        if random.random() <= self.epsilon:
            action = random.choice([0, 1])
        else:
            q = self.dqn_model.predict(self.input_image)
            action = np.argmax(q)
        # train the model if demand
        loss = 0
        if self.mode == 'train' and self.num_iters > self.num_observes:
            minibatch = random.sample(self.replay_memory_record, self.batch_size)
            states, actions, rewards, states1, is_game_running = zip(*minibatch)
            states = np.concatenate(states)
            states1 = np.concatenate(states1)
            targets = self.dqn_model.predict(states1)
            targets[range(self.batch_size), actions] = rewards + self.discount_factor * np.max(self.dqn_model.predict(states1), axis=1) * is_game_running
            loss = self.dqn_model.train_on_batch(states, targets)
            if self.num_iters % self.save_interval == 0:
                self.saveModel(self.backuppath)
        # print some infos
        if self.mode == 'train':
            print('STATE: train, ITER: %s, EPSILON: %s, ACTION: %s, REWARD: %s, LOSS: %s, MAX_SCORE: %s' % (self.num_iters, self.epsilon, action, reward, float(loss), self.max_score))
        else:
            print('STATE: test, ACTION: %s, MAX_SCORE: %s' % (action, self.max_score))
        return action
    '''load model'''
    def loadModel(self, modelpath):
        if self.mode == 'train':
            print('[INFO]: load checkpoints from %s and %s' % (modelpath, modelpath.replace('h5', 'pkl')))
            self.dqn_model.load_weights(modelpath)
            data_dict = pickle.load(open(modelpath.replace('h5', 'pkl'), 'rb'))
            self.max_score = data_dict['max_score']
            self.epsilon = data_dict['epsilon']
            self.num_iters = data_dict['num_iters']
            self.replay_memory_record = data_dict['replay_memory_record']
        else:
            print('[INFO]: load checkpoints from %s' % modelpath)
            self.dqn_model.load_weights(modelpath)
            self.max_score = 0
            self.epsilon = self.final_epsilon
            self.num_iters = 0
            self.replay_memory_record = deque()
    '''save model'''
    def saveModel(self, modelpath):
        self.dqn_model.save_weights(modelpath, overwrite=True)
        data_dict = {
                        'num_iters': self.num_iters,
                        'epsilon': self.epsilon,
                        'replay_memory_record': self.replay_memory_record,
                        'max_score': self.max_score
                    }
        with open(modelpath.replace('h5', 'pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        print('[INFO]: save checkpoints into %s and %s' % (modelpath, modelpath.replace('h5', 'pkl')))
    '''record the necessary information'''
    def record(self, action, reward, score, is_game_running, image):
        # preprocess game frames
        image = self.preprocess(image, self.imagesize)
        # record the scene and corresponding info
        if self.input_image is None:
            self.input_image = np.stack((image,)*self.num_input_frames, axis=2)
            self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
        else:
            image = image.reshape(1, image.shape[0], image.shape[1], 1)
            next_input_image = np.append(image, self.input_image[:, :, :, :self.num_input_frames-1], axis=3)
            self.replay_memory_record.append((self.input_image, np.array([action]), np.array([reward]), next_input_image, np.array([int(is_game_running)])))
            self.input_image = next_input_image
        if len(self.replay_memory_record) > self.replay_memory_size:
            self.replay_memory_record.popleft()
        # record the max score so far
        if score > self.max_score:
            self.max_score = score
    '''preprocess the image'''
    def preprocess(self, image, imagesize):
        image = skimage.color.rgb2gray(image)
        image = skimage.transform.resize(image, imagesize, mode='constant')
        image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
        image = image / 255.0
        return image