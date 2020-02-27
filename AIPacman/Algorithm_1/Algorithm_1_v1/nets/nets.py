'''
Function:
	define the network
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import sys
import time
import torch
import random
import numpy as np
import torch.nn as nn
from collections import deque


'''dqn'''
class DQNet(nn.Module):
	def __init__(self, config, **kwargs):
		super(DQNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=config.num_element_types*config.num_continuous_frames, out_channels=16, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.fc1 = nn.Linear(in_features=config.frame_size[0]*config.frame_size[1]*32, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=4)
		self.relu = nn.ReLU(inplace=True)
		self.__initWeights()
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x).view(x.size(0), -1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x
	def __initWeights(self):
		nn.init.normal_(self.conv1.weight, std=0.01)
		nn.init.normal_(self.conv2.weight, std=0.01)
		nn.init.normal_(self.fc1.weight, std=0.01)
		nn.init.normal_(self.fc2.weight, std=0.01)
		nn.init.constant_(self.conv1.bias, 0.1)
		nn.init.constant_(self.conv2.bias, 0.1)
		nn.init.constant_(self.fc1.bias, 0.1)
		nn.init.constant_(self.fc2.bias, 0.1)


'''dqn agent'''
class DQNAgent():
	def __init__(self, game_pacman_agent, dqn_net, config, **kwargs):
		self.game_pacman_agent = game_pacman_agent
		self.dqn_net = dqn_net
		self.config = config
		self.game_memories = deque()
		self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
	'''train'''
	def train(self):
		# prepare
		if not os.path.exists(self.config.save_dir):
			os.mkdir(self.config.save_dir)
		if self.config.use_cuda:
			self.dqn_net = self.dqn_net.cuda()
		FloatTensor = torch.cuda.FloatTensor if self.config.use_cuda else torch.FloatTensor
		# GoGoGo
		frames = []
		optimizer = torch.optim.Adam(self.dqn_net.parameters())
		num_iter = 0
		image = None
		image_prev = None
		action_pred = None
		score_best = 0
		num_games = 0
		num_wins = 0
		while True:
			if len(self.game_memories) > self.config.max_memory_size:
				self.game_memories.popleft()
			frame, is_win, is_gameover, reward, action = self.game_pacman_agent.nextFrame(action=action_pred)
			score_best = max(self.game_pacman_agent.score, score_best)
			if is_gameover:
				self.game_pacman_agent.reset()
				if len(self.game_memories) >= self.config.max_explore_iterations:
					num_games += 1
					num_wins += int(is_win)
			frames.append(frame)
			if len(frames) == self.config.num_continuous_frames:
				image_prev = image
				image = np.concatenate(frames, -1)
				exprience = (image, image_prev, reward, self.formatAction(action, outformat='networkformat'), is_gameover)
				frames.pop(0)
				if image_prev is not None:
					self.game_memories.append(exprience)
			# explore
			if len(self.game_memories) < self.config.max_explore_iterations:
				self.__logging('[STATE]: explore, [MEMORYLEN]: %d' % len(self.game_memories), self.config.logfile)
			# train
			else:
				# --get data
				num_iter += 1
				images_input = []
				images_prev_input = []
				is_gameovers = []
				actions = []
				rewards = []
				for each in random.sample(self.game_memories, self.config.batch_size):
					image_input = each[0].astype(np.float32)
					image_input.resize((1, *image_input.shape))
					images_input.append(image_input)
					image_prev_input = each[1].astype(np.float32)
					image_prev_input.resize((1, *image_prev_input.shape))
					images_prev_input.append(image_prev_input)
					rewards.append(each[2])
					actions.append(each[3])
					is_gameovers.append(each[4])
				images_input_torch = torch.from_numpy(np.concatenate(images_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)
				images_prev_input_torch = torch.from_numpy(np.concatenate(images_prev_input, 0)).permute(0, 3, 1, 2).type(FloatTensor)
				# --compute loss
				optimizer.zero_grad()
				q_t = self.dqn_net(images_input_torch)
				q_t = torch.max(q_t, dim=1)[0]
				loss = self.mse_loss(torch.Tensor(rewards).type(FloatTensor) + (1 - torch.Tensor(is_gameovers).type(FloatTensor)) * (0.95 * q_t),
									 (self.dqn_net(images_prev_input_torch) * torch.Tensor(actions).type(FloatTensor)).sum(1))
				loss.backward()
				optimizer.step()
				# --make decision
				prob = max(self.config.eps_start-(self.config.eps_start-self.config.eps_end)/self.config.eps_num_steps*num_iter, self.config.eps_end)
				if random.random() > prob:
					with torch.no_grad():
						self.dqn_net.eval()
						image_input = image.astype(np.float32)
						image_input.resize((1, *image_input.shape))
						image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
						action_pred = self.dqn_net(image_input_torch).view(-1).tolist()
						action_pred = self.formatAction(action_pred, outformat='oriactionformat')
						self.dqn_net.train()
				else:
					action_pred = None
				self.__logging('[STATE]: training, [ITER]: %d, [LOSS]: %.3f, [ACTION]: %s, [BEST SCORE]: %d, [NUMWINS/NUMGAMES]: %d/%d' % (num_iter, loss.item(), str(action_pred), score_best, num_wins, num_games), self.config.logfile)
				if num_iter % self.config.save_interval == 0 or num_iter == self.config.max_train_iterations:
					torch.save(self.dqn_net.state_dict(), os.path.join(self.config.save_dir, '%s.pkl' % num_iter))
				if num_iter == self.config.max_train_iterations:
					self.__logging('Train Finished!', self.config.logfile)
					sys.exit(-1)
	'''test'''
	def test(self):
		if self.config.use_cuda:
			self.dqn_net = self.dqn_net.cuda()
		self.dqn_net.eval()
		FloatTensor = torch.cuda.FloatTensor if self.config.use_cuda else torch.FloatTensor
		frames = []
		action_pred = None
		while True:
			frame, is_win, is_gameover, reward, action = self.game_pacman_agent.nextFrame(action=action_pred)
			if is_gameover:
				self.game_pacman_agent.reset()
			frames.append(frame)
			if len(frames) == self.config.num_continuous_frames:
				image = np.concatenate(frames, -1)
				if random.random() > self.config.eps_end:
					with torch.no_grad():
						image_input = image.astype(np.float32)
						image_input.resize((1, *image_input.shape))
						image_input_torch = torch.from_numpy(image_input).permute(0, 3, 1, 2).type(FloatTensor)
						action_pred = self.dqn_net(image_input_torch).view(-1).tolist()
						action_pred = self.formatAction(action_pred, outformat='oriactionformat')
				else:
					action_pred = None
				frames.pop(0)
			print('[ACTION]: %s' % str(action_pred))
	'''format action'''
	def formatAction(self, action, outformat='networkformat'):
		if outformat == 'networkformat':
			# left
			if action == [-1, 0]:
				return [1, 0, 0, 0]
			# right
			elif action == [1, 0]:
				return [0, 1, 0, 0]
			# up
			elif action == [0, -1]:
				return [0, 0, 1, 0]
			# down
			elif action == [0, 1]:
				return [0, 0, 0, 1]
			# error
			else:
				raise RuntimeError('something wrong in DQNAgent.formatAction')
		elif outformat == 'oriactionformat':
			idx = action.index(max(action))
			# left
			if idx == 0:
				return [-1, 0]
			# right
			elif idx == 1:
				return [1, 0]
			# up
			elif idx == 2:
				return [0, -1]
			# down
			elif idx == 3:
				return [0, 1]
			# error
			else:
				raise RuntimeError('something wrong in DQNAgent.formatAction')
		else:
			raise ValueError('DQNAgent.formatAction unsupport outformat %s...' % outformat)
	def __logging(self, message, savefile=None):
		content = '%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)
		if savefile:
			f = open(savefile, 'a')
			f.write(content + '\n')
			f.close()
		print(content)