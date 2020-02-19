'''
Function:
    Define the q learning algorithm
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import pickle
import random
import numpy as np


'''q learning agent'''
class QLearningAgent():
    def __init__(self, mode, **kwargs):
        self.mode = mode
        # learning rate
        self.learning_rate = 0.7
        # discount factor(also named discount rate)
        self.discount_factor = 0.95
        # store the necessary history data, the format is [previous_state, previous_action, state, reward]
        self.history_storage = []
        # store the q values, the last dimension is [value_for_do_nothing, value_for_flappy]
        self.qvalues_storage = np.zeros((130, 130, 20, 2))
        # store the score for each episode
        self.scores_storage = []
        # previous state
        self.previous_state = []
        # 0 means do nothing, 1 means flappy
        self.previous_action = 0
        # number of episode
        self.num_episode = 0
        # the max score so far
        self.max_score = 0
    '''make a decision'''
    def act(self, delta_x, delta_y, bird_speed):
        if not self.previous_state:
            self.previous_state = [delta_x, delta_y, bird_speed]
            return self.previous_action
        if self.mode == 'train':
            state = [delta_x, delta_y, bird_speed]
            self.history_storage.append([self.previous_state, self.previous_action, state, 0])
            self.previous_state = state
        # make a decision according to the qvalues
        if self.qvalues_storage[delta_x, delta_y, bird_speed][0] >= self.qvalues_storage[delta_x, delta_y, bird_speed][1]:
            self.previous_action = 0
        else:
            self.previous_action = 1
        return self.previous_action
    '''set reward'''
    def setReward(self, reward):
        if self.history_storage:
            self.history_storage[-1][3] = reward
    '''update the qvalues_storage after an episode'''
    def update(self, score, is_logging=True):
        self.num_episode += 1
        self.max_score = max(self.max_score, score)
        self.scores_storage.append(score)
        if is_logging:
            print('Episode: %s, Score: %s, Max Score: %s' % (self.num_episode, score, self.max_score))
        if self.mode == 'train':
            history = list(reversed(self.history_storage))
            # penalize last num_penalization states before crash
            num_penalization = 2 
            for item in history:
                previous_state, previous_action, state, reward = item
                if num_penalization > 0:
                    num_penalization -= 1
                    reward = -1000000
                x_0, y_0, z_0 = previous_state
                x_1, y_1, z_1 = state
                self.qvalues_storage[x_0, y_0, z_0, previous_action] = (1 - self.learning_rate) * self.qvalues_storage[x_0, y_0, z_0, previous_action] +\
                                                                       self.learning_rate * (reward + self.discount_factor * max(self.qvalues_storage[x_1, y_1, z_1]))
            self.history_storage = []
    '''save the model'''
    def saveModel(self, modelpath):
        data = {
                'num_episode': self.num_episode,
                'max_score': self.max_score,
                'scores_storage': self.scores_storage,
                'qvalues_storage': self.qvalues_storage
            }
        with open(modelpath, 'wb') as f:
            pickle.dump(data, f)
        print('[INFO]: save checkpoints in %s...' % modelpath)
    '''load the model'''
    def loadModel(self, modelpath):
        print('[INFO]: load checkpoints from %s...' % modelpath)
        with open(modelpath, 'rb') as f:
            data = pickle.load(f)
        self.num_episode = data.get('num_episode')
        self.qvalues_storage = data.get('qvalues_storage')


'''q learning agent with ε-greedy policy'''
class QLearningGreedyAgent(QLearningAgent):
    def __init__(self, mode, **kwargs):
        super(QLearningGreedyAgent, self).__init__(mode, **kwargs)
        self.epsilon = 0.1
        self.epsilon_end = 0.0
        self.epsilon_decay = 1e-5
    '''make a decision'''
    def act(self, delta_x, delta_y, bird_speed):
        if not self.previous_state:
            self.previous_state = [delta_x, delta_y, bird_speed]
            return self.previous_action
        if self.mode == 'train':
            state = [delta_x, delta_y, bird_speed]
            self.history_storage.append([self.previous_state, self.previous_action, state, 0])
            self.previous_state = state
            # greedy policy
            if random.random() <= self.epsilon:
                self.previous_action = random.choice([0, 1])
            else:
                if self.qvalues_storage[delta_x, delta_y, bird_speed][0] >= self.qvalues_storage[delta_x, delta_y, bird_speed][1]:
                    self.previous_action = 0
                else:
                    self.previous_action = 1
            return self.previous_action
        else:
            super().act(delta_x, delta_y, bird_speed)
    '''update the qvalues_storage after an episode'''
    def update(self, score, is_logging=True):
        self.num_episode += 1
        self.max_score = max(self.max_score, score)
        self.scores_storage.append(score)
        if is_logging:
            print('Episode: %s, Epsilon: %s, Score: %s, Max Score: %s' % (self.num_episode, self.epsilon, score, self.max_score))
        if self.mode == 'train':
            history = list(reversed(self.history_storage))
            # penalize last num_penalization states before crash
            num_penalization = 2 
            for item in history:
                previous_state, previous_action, state, reward = item
                if num_penalization > 0:
                    num_penalization -= 1
                    reward = -1000000
                x_0, y_0, z_0 = previous_state
                x_1, y_1, z_1 = state
                self.qvalues_storage[x_0, y_0, z_0, previous_action] = (1 - self.learning_rate) * self.qvalues_storage[x_0, y_0, z_0, previous_action] +\
                                                                       self.learning_rate * (reward + self.discount_factor * max(self.qvalues_storage[x_1, y_1, z_1]))
            self.history_storage = []
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_decay
    '''save the model'''
    def saveModel(self, modelpath):
        data = {
                'num_episode': self.num_episode,
                'max_score': self.max_score,
                'scores_storage': self.scores_storage,
                'qvalues_storage': self.qvalues_storage,
                'epsilon': self.epsilon
            }
        with open(modelpath, 'wb') as f:
            pickle.dump(data, f)
        print('[INFO]: save checkpoints in %s...' % modelpath)
    '''load the model'''
    def loadModel(self, modelpath):
        print('[INFO]: load checkpoints from %s...' % modelpath)
        with open(modelpath, 'rb') as f:
            data = pickle.load(f)
        self.num_episode = data.get('num_episode')
        self.qvalues_storage = data.get('qvalues_storage')
        self.epsilon = data.get('epsilon')