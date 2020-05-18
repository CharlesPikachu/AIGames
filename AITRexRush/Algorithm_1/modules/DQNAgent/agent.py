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
	def __init__(self, mode, fps, checkpointspath, **kwargs):
		self.mode = mode
		self.fps = fps
		self.checkpointspath = checkpointspath
		# define the necessary variables
		self.imagesize = (84, 84)
		self.num_input_frames = 4
		self.num_actions = 3
		self.save_interval = 5000
		self.replay_memory_record = deque()
		self.init_epsilon = 0.1
		self.end_epsilon = 1e-4
		self.epsilon = self.init_epsilon
		self.batch_size = 32
		self.replay_memory_size = 1e4
		self.discount_factor = 0.99
		self.pos_save_prob = 0.1
		self.num_observes = 3200
		self.num_explores = 1e5
		self.input_image = None
		self.num_iters = 0
		self.num_games = 0
		self.score = 0
		self.max_score = 0
		self.use_cuda = torch.cuda.is_available()
		self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		# define the model
		self.dqn_model = DeepQNetwork(self.imagesize, self.num_input_frames, self.num_actions)
		self.dqn_model = self.dqn_model.cuda() if self.use_cuda else self.dqn_model
		self.dqn_model.apply(DeepQNetwork.initWeights)
		self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=1e-4)
		self.loss_func = nn.MSELoss()
	'''train the agent'''
	def train(self, game_cotroller):
		action = np.array([0] * self.num_actions)
		action[0] = 1
		image, score, is_dead = game_cotroller.run(action)
		image = self.preprocess(image, self.imagesize)
		self.input_image = np.tile(image, (self.num_input_frames, 1, 1))
		self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
		last_time = 0
		while True:
			# randomly or use dqn_model to decide the action of T-Rex
			action = np.array([0] * self.num_actions)
			if random.random() <= self.epsilon:
				action[random.choice(list(range(self.num_actions)))] = 1
			else:
				self.dqn_model.eval()
				input_image = torch.from_numpy(self.input_image).type(self.FloatTensor)
				with torch.no_grad():
					preds = self.dqn_model(input_image).cpu().data.numpy()
				action[np.argmax(preds)] = 1
				self.dqn_model.train()
			# perform the action
			image, score, is_dead = game_cotroller.run(action)
			image = self.preprocess(image, self.imagesize)
			image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
			input_image_prev = self.input_image.copy()
			self.input_image = np.append(image, self.input_image[:, :self.num_input_frames-1, :, :], axis=1)
			# control the FPS
			if last_time:
				fps_now = 1 / (time.time() - last_time)
				if fps_now > self.fps:
					time.sleep(1 / self.fps - 1 / fps_now)
			last_time = time.time()
			# get reward
			if is_dead:
				self.num_games += 1
				reward = -1
			else:
				reward = 0.1
			# record score
			self.score = score
			if score > self.max_score:
				self.max_score = score
			# save the game data for training dqn
			if is_dead or random.random() <= self.pos_save_prob:
				self.replay_memory_record.append([input_image_prev, self.input_image, action, np.array([int(is_dead)]), np.array([reward])])
			if len(self.replay_memory_record) > self.replay_memory_size:
				self.replay_memory_record.popleft()
			# train the model
			loss = torch.Tensor([0]).type(self.FloatTensor)
			if self.num_iters > self.num_observes:
				self.optimizer.zero_grad()
				minibatch = random.sample(self.replay_memory_record, self.batch_size)
				states, states1, actions, is_deads, rewards = zip(*minibatch)
				states = torch.from_numpy(np.concatenate(states)).type(self.FloatTensor)
				states1 = torch.from_numpy(np.concatenate(states1)).type(self.FloatTensor)
				actions = torch.from_numpy(np.concatenate(actions)).type(self.FloatTensor).view(self.batch_size, -1)
				is_deads = torch.from_numpy(np.concatenate(is_deads)).type(self.FloatTensor)
				rewards = torch.from_numpy(np.concatenate(rewards)).type(self.FloatTensor)
				with torch.no_grad():
					targets = rewards + self.discount_factor * self.dqn_model(states1).max(-1)[0] * (1 - is_deads)
					targets = targets.detach()
				preds = torch.sum(self.dqn_model(states) * actions, dim=1)
				loss = self.loss_func(preds, targets)
				loss.backward()
				self.optimizer.step()
			# update epsilon
			self.num_iters += 1
			if (self.epsilon > self.end_epsilon) and (self.num_iters > self.num_observes):
				self.epsilon -= (self.init_epsilon - self.end_epsilon) / self.num_explores
			# save the model
			if self.num_iters % self.save_interval == 0:
				self.save(self.checkpointspath)
			# print necessary info
			print('[State]: train, [Games]: %s, [Iter]: %s, [Score]: %s, [Max Score]: %s, [Epsilon]: %s, [Action]: %s, [Reward]: %s, [Loss]: %.3f' %  (self.num_games, self.num_iters, self.score, self.max_score, self.epsilon, np.argmax(action), reward, loss.item()))
	'''test the agent'''
	def test(self, game_cotroller):
		action = np.array([0] * self.num_actions)
		action[0] = 1
		image, score, is_dead = game_cotroller.run(action)
		image = self.preprocess(image, self.imagesize)
		self.input_image = np.tile(image, (self.num_input_frames, 1, 1))
		self.input_image = self.input_image.reshape(1, self.input_image.shape[0], self.input_image.shape[1], self.input_image.shape[2])
		last_time = 0
		while True:
			# randomly or use dqn_model to decide the action of T-Rex
			action = np.array([0] * self.num_actions)
			if random.random() <= self.end_epsilon:
				action[random.choice(list(range(self.num_actions)))] = 1
			else:
				self.dqn_model.eval()
				input_image = torch.from_numpy(self.input_image).type(self.FloatTensor)
				with torch.no_grad():
					preds = self.dqn_model(input_image).cpu().data.numpy()
				action[np.argmax(preds)] = 1
			# perform the action
			image, score, is_dead = game_cotroller.run(action)
			image = self.preprocess(image, self.imagesize)
			image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
			self.input_image = np.append(image, self.input_image[:, :self.num_input_frames-1, :, :], axis=1)
			if is_dead: self.num_games += 1
			# control the FPS
			if last_time:
				fps_now = 1 / (time.time() - last_time)
				if fps_now > self.fps:
					time.sleep(1 / self.fps - 1 / fps_now)
			last_time = time.time()
			# record score
			self.score = score
			if score > self.max_score:
				self.max_score = score
			# print necessary info
			print('[State]: test, [Games]: %s, [Score]: %s, [Max Score]: %s, [Epsilon]: %s, [Action]: %s' %  (self.num_games, self.score, self.max_score, self.end_epsilon, np.argmax(action)))
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
		image = cv2.resize(image, size)
		image[image > 0] = 255
		image = np.expand_dims(image, 0)
		return image