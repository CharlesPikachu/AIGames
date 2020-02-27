'''
Function:
    define the dqn agent
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import pickle
import random
import skimage
import numpy as np
import skimage.color
import torch.nn as nn
import skimage.exposure
import skimage.transform
from collections import deque
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
        self.use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.batch_size = 32
        # the instanced model
        self.dqn_model = deepQNetwork(imagesize=self.imagesize, in_channels=self.num_input_frames, num_actions=self.num_actions)
        self.dqn_model = self.dqn_model.cuda() if self.use_cuda else self.dqn_model
        # define the optimizer and loss function if mode is train
        if mode == 'train':
            self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=1e-4)
            self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
    '''get the next action'''
    def nextAction(self, reward):
        # some necessary update
        if self.epsilon > self.final_epsilon and self.num_iters > self.num_observes:
            self.epsilon -= (self.init_epsilon - self.final_epsilon) / self.num_explores
        self.num_iters += 1
        # make decision
        if self.num_iters < self.num_observes or random.random() <= self.epsilon:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                x = torch.from_numpy(self.input_image).type(self.FloatTensor)
                preds = self.dqn_model(x).view(-1)
                action = preds.argmax().item()
        # train the model if demand
        loss = torch.tensor([0])
        if self.mode == 'train' and self.num_iters > self.num_observes:
            self.optimizer.zero_grad()
            minibatch = random.sample(self.replay_memory_record, self.batch_size)
            states, actions, rewards, states1, is_gameovers = zip(*minibatch)
            states = torch.from_numpy(np.concatenate(states, axis=0)).type(self.FloatTensor)
            actions = torch.from_numpy(np.concatenate(actions, axis=0)).type(self.FloatTensor).view(self.batch_size, self.num_actions)
            rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).type(self.FloatTensor).view(self.batch_size)
            states1 = torch.from_numpy(np.concatenate(states1, axis=0)).type(self.FloatTensor)
            is_gameovers = torch.from_numpy(np.concatenate(is_gameovers, axis=0)).type(self.FloatTensor).view(self.batch_size)
            q_t = self.dqn_model(states1)
            q_t = torch.max(q_t, dim=1)[0]
            loss = self.mse_loss(rewards + (1 - is_gameovers) * (self.discount_factor * q_t),
                                 (self.dqn_model(states) * actions).sum(1))
            loss.backward()
            self.optimizer.step()
            if self.num_iters % self.save_interval == 0:
                self.saveModel(self.backuppath)
        # print some infos
        if self.mode == 'train':
            print('STATE: train, ITER: %s, EPSILON: %s, ACTION: %s, REWARD: %s, LOSS: %s, MAX_SCORE: %s' % (self.num_iters, self.epsilon, action, reward, loss.item(), self.max_score))
        else:
            print('STATE: test, ACTION: %s, MAX_SCORE: %s' % (action, self.max_score))
        return action
    '''load model'''
    def loadModel(self, modelpath):
        print('[INFO]: load checkpoints from %s and %s' % (modelpath, modelpath.replace('pth', 'pkl')))
        model_dict = torch.load(modelpath)
        data_dict = pickle.load(open(modelpath.replace('pth', 'pkl'), 'rb'))
        self.dqn_model.load_state_dict(model_dict['model'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.max_score = data_dict['max_score']
        self.epsilon = data_dict['epsilon'] if self.mode == 'train' else self.final_epsilon
        self.num_iters = data_dict['num_iters'] if self.mode == 'train' else self.num_observes + 1
        self.replay_memory_record = data_dict['replay_memory_record']
    '''save model'''
    def saveModel(self, modelpath):
        model_dict = {
                        'model': self.dqn_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
        torch.save(model_dict, modelpath)
        data_dict = {
                        'num_iters': self.num_iters,
                        'epsilon': self.epsilon,
                        'replay_memory_record': self.replay_memory_record,
                        'max_score': self.max_score
                    }
        with open(modelpath.replace('pth', 'pkl'), 'wb') as f:
            pickle.dump(data_dict, f)
        print('[INFO]: save checkpoints into %s and %s' % (modelpath, modelpath.replace('pth', 'pkl')))
    '''record the necessary information'''
    def record(self, action, reward, score, is_game_running, image):
        # preprocess game frames
        image = self.preprocess(image, self.imagesize)
        # record the scene and corresponding info
        if self.input_image is None:
            self.input_image = np.stack((image,)*self.num_input_frames, axis=2)
            self.input_image = np.transpose(self.input_image, (2, 0, 1))
            self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
        else:
            image = image.reshape(1, 1, image.shape[0], image.shape[1])
            next_input_image = np.append(image, self.input_image[:, :self.num_input_frames-1, :, :], axis=1)
            action = [0, 1] if action else [1, 0]
            self.replay_memory_record.append((self.input_image, np.array(action), np.array([reward]), next_input_image, np.array([int(not is_game_running)])))
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