'''
Function:
	dqn agent
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import time
import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque
from .network import DeepQNetwork


'''dqn agent'''
class DQNAgent():
	def __init__(self, mode, checkpointspath, **kwargs):
		self.mode = mode
		self.checkpointspath = checkpointspath
		# define the necessary variables
		self.imagesize = (96, 96)
		self.num_input_frames = 4
		self.num_actions = 2
		self.save_interval = 5000
		self.replay_memory_record = deque()
		self.epsilon = 0.1
		self.init_epsilon = 0.1
		self.end_epsilon = 1e-4
		self.batch_size = 32
		self.replay_memory_size = 2e4
		self.discount_factor = 0.99
		self.num_observes = 3200
		self.num_explores = 3e6
		self.input_image = None
		self.num_iters = 0
		self.score = 0
		self.max_score = 0
		self.use_cuda = torch.cuda.is_available()
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		# define the model
		self.dqn_model = DeepQNetwork(self.imagesize, self.num_input_frames, self.num_actions)
		self.dqn_model = self.dqn_model.cuda() if self.use_cuda else self.dqn_model
		self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=1e-4)
		self.loss_func = nn.MSELoss(reduction='none')
	'''train the agent'''
	def train(self, game_cotroller):
		action = np.array([0] * self.num_actions)
		action[0] = 1
		image, score, is_dead = game_cotroller.run(action)
		image = self.preprocess(image, self.imagesize)
		self.input_image = np.stack((image,)*self.num_input_frames, axis=2)
		self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
		while True:
			# randomly or use dqn_model to decide the action of T-Rex
			action = np.array([0] * self.num_actions)
			if random.random() <= self.epsilon:
				action[random.choice(list(range(self.num_actions)))] = 1
			else:
				self.dqn_model.eval()
				input_image = torch.from_numpy(self.input_image).type(self.FloatTensor)
				input_image = input_image.permute(0, 3, 1, 2).contiguous()
				preds = self.dqn_model(input_image).cpu().data.numpy()
				action[np.argmax(preds)] = 1
				self.dqn_model.train()
			# perform the action
			image, score, is_dead = game_cotroller.run(action)
			image = self.preprocess(image, self.imagesize)
			image = image.reshape(1, image.shape[0], image.shape[1], 1)
			input_image_prev = self.input_image.copy()
			self.input_image = np.append(image, self.input_image[:, :, :, :self.num_input_frames-1], axis=3)
			# get reward
			if is_dead:
				reward = -1
			else:
				reward = 0.1
			# record score
			self.score = score
			if score > self.max_score:
				self.max_score = score
			# save the game data for training dqn
			self.replay_memory_record.append([input_image_prev, self.input_image, np.array([np.argmax(action)]), np.array([int(is_dead)]), np.array([reward])])
			if len(self.replay_memory_record) > self.replay_memory_size:
				self.replay_memory_record.popleft()
			# train the model
			loss = torch.Tensor([0]).type(self.FloatTensor)
			if self.num_iters > self.num_observes:
				self.optimizer.zero_grad()
				minibatch = random.sample(self.replay_memory_record, self.batch_size)
				states, states1, actions, is_deads, rewards = zip(*minibatch)
				states = torch.from_numpy(np.concatenate(states)).type(self.FloatTensor)
				states = states.permute(0, 3, 1, 2).contiguous()
				states1 = torch.from_numpy(np.concatenate(states1)).type(self.FloatTensor)
				states1 = states1.permute(0, 3, 1, 2).contiguous()
				actions = np.concatenate(actions)
				is_deads = np.concatenate(is_deads)
				rewards = np.concatenate(rewards)
				targets = self.dqn_model(states1).cpu().data.numpy()
				targets[range(self.batch_size), actions] = rewards + self.discount_factor * np.max(self.dqn_model(states1).cpu().data.numpy(), axis=1) * (1 - is_deads)
				targets = torch.from_numpy(targets).type(self.FloatTensor)
				loss = self.loss_func(self.dqn_model(states), targets).sum() / self.batch_size
				loss.backward()
				self.optimizer.step()
			# save the model
			if self.num_iters % self.save_interval == 0:
				self.save(self.checkpointspath)
			# update epsilon
			self.num_iters += 1
			if (self.epsilon > self.end_epsilon) and (self.num_iters > self.num_observes):
				self.epsilon -= (self.init_epsilon - self.end_epsilon) / self.num_explores
			# print necessary info
			print('[State]: train, [Iter]: %s, [Epsilon]: %s, [Action]: %s, [Reward]: %s, [Loss]: %s, [Score]: %s, [Max Score]: %s ' %  (self.num_iters, self.epsilon, np.argmax(action), reward, loss.item(), self.score, self.max_score))
	'''test the agent'''
	def test(self, game_cotroller):
		pass
	'''load checkpoints'''
	def load(self, checkpointspath):
		print('Loading checkpoints from %s...' % checkpointspath)
		self.dqn_model.load_state_dict(torch.load(checkpointspath))
	'''save checkpoints'''
	def save(self, checkpointspath):
		print('Saving checkpoints into %s...' % checkpointspath)
		torch.save(self.dqn_model.state_dict(), checkpointspath)
	'''preprocess image'''
	def preprocess(self, image, size):
		image = np.array(image)
		image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
		return image