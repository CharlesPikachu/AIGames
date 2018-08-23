# 作者: Charles
# 公众号: Charles的皮卡丘
# 网络模型
# for model_1
import os
import json
import keras
import random
from collections import deque
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Activation, Conv2D, Flatten, Dense, MaxPooling2D
import sys
sys.path.append('..')
from utils.utils import *
'''
# for model_2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
'''


'''
Function:
	模型一: Keras框架搭建, 简单测试用
'''
class model_1():
	def __init__(self, options, with_pool=False):
		if with_pool:
			self.model = self.build_pool_model(options)
		else:
			self.model = self.build_model(options)
		self.save_model_info()
	# 创建模型
	def build_model(self, options):
		print('[INFO]: Start to build model_1 without pool...')
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=options['input_shape']))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (4, 4), strides=(1, 1), padding='same'))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dense(options['num_actions']))
		optim = Adam(lr=options['lr'])
		model.compile(loss='mse', optimizer=optim)
		return model
	# 创建带池化层的model
	def build_pool_model(self, options):
		print('[INFO]: Start to build model_1 with pool...')
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=options['input_shape']))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dense(options['num_actions']))
		optim = Adam(lr=options['lr'])
		model.compile(loss='mse', optimizer=optim)
		return model
	# 模型训练
	def train(self, agent, options, batch_idx=0):
		Data_deque = deque()
		height = options['input_shape'][0]
		width = options['input_shape'][1]
		num_channels = options['input_shape'][2]
		# 小恐龙跳跃的概率
		start_prob_jump = options['start_prob_jump']
		end_prob_jump = options['end_prob_jump']
		interval_prob = options['interval_prob']
		prob = start_prob_jump
		# 如果需要继续训练，这里调整prob初始值
		# prob = start_prob_jump - (start_prob_jump - end_prob_jump) / interval_prob * 20000
		# 操作num_operations后进行训练
		num_operations = options['num_operations']
		# (1, 0) -> do nothing
		# (0, 1) -> jump
		actions = np.zeros(options['num_actions'])
		img, score, is_dead = agent.perform_operation(actions)
		size = (width, height)
		img = preprocess_img(img, size=size, use_canny=options['use_canny'])
		# 模型训练的输入
		x_now = np.stack((img,) * num_channels, axis=2).reshape(1, height, width, num_channels)
		x_init = x_now
		loss_dict = {}
		i = 0
		while True:
			i += 1
			actions = np.zeros(options['num_actions'])
			if random.random() <= prob:
				print('[INFO]: Dino actions randomly...')
				action_idx = random.randint(0, len(actions)-1)
				actions[action_idx] = 1
			else:
				print('[INFO]: Dino actions controlled by network...')
				Q_now = self.model.predict(x_now)
				action_idx = np.argmax(Q_now)
				actions[action_idx] = 1
			img, score, is_dead = agent.perform_operation(actions)
			img = preprocess_img(img, size=size, use_canny=options['use_canny'])
			reward = self.score2reward(score, is_dead, actions)
			img = img.reshape(1, height, width, 1)
			x_next = np.append(img, x_now[:, :, :, :num_channels-1], axis=3)
			Data_deque.append((x_now, action_idx, reward, x_next, is_dead, score))
			if len(Data_deque) > options['data_memory']:
				Data_deque.popleft()
				# Data_deque = deque()
			'''
			if len(Data_deque) == num_operations:
				agent.pause()
				save_dict(Data_deque, options['data_dir'], 'Data_deque_%d.pkl' % batch_idx)
				data_len = len(Data_deque) // options['num_sampling']
				for i in range(options['num_sampling']):
					batch_idx += 1
					print('[INFO]: Start to train <Batch-%d>...' % batch_idx)
					start = i * data_len
					end = (i+1) * data_len
					data = deque()
					for j in range(start, end):
						data.append(Data_deque[j])
					loss = self.trainBatch(random.sample(data, options['batch_size']), options)
					loss_dict[batch_idx] = loss
					print('\t<Loss>: %.2f' % loss)
					if batch_idx % options['save_interval'] == 0:
						if not os.path.exists(options['savepath']):
							os.mkdir(options['savepath'])
						savename = options['savename'] + '_' + str(batch_idx) + '.h5'
						self.model.save_weights(os.path.join(options['savepath'], savename))
						save_dict(loss_dict, options['log_dir'], 'loss.pkl')
					if batch_idx == options['max_batch']:
						break
				if batch_idx == options['max_batch']:
					break
			agent.resume()
			'''
			if i > num_operations:
				# i = 0 if len(Data_deque) < 1000 else i
				# i = 0
				# agent.pause()
				batch_idx += 1
				print('[INFO]: Start to train <Batch-%d>...' % batch_idx)
				loss = self.trainBatch(random.sample(Data_deque, options['batch_size']), options)
				loss_dict[batch_idx] = loss
				# print('\t<Loss>: %.3f' % loss)
				print('\t<Loss>: %.3f, <Action>: %d' % (loss, action_idx))
				if batch_idx % options['save_interval'] == 0:
					if not os.path.exists(options['savepath']):
						os.mkdir(options['savepath'])
					savename = options['savename'] + '_' + str(batch_idx) + '.h5'
					self.model.save_weights(os.path.join(options['savepath'], savename))
					save_dict(loss_dict, options['log_dir'], 'loss.pkl')
					if (len(Data_deque) == options['data_memory']) and (batch_idx % 5000 == 0):
						save_dict(Data_deque, options['data_dir'], 'Data_deque_%d.pkl' % batch_idx)
				if batch_idx == options['max_batch']:
					break
				# agent.resume()
			x_now = x_init if is_dead else x_next
			# 逐渐减小人为设定的控制，让网络自己决定如何行动
			if prob > end_prob_jump and i > num_operations:
				prob -= (start_prob_jump - end_prob_jump) / interval_prob
		savename = options['savename'] + '_' + str(batch_idx) + '.h5'
		self.model.save_weights(os.path.join(options['savepath'], savename))
	# 训练一个Batch数据
	def trainBatch(self, data_batch, options):
		height = options['input_shape'][0]
		width = options['input_shape'][1]
		num_channels = options['input_shape'][2]
		inputs = np.zeros((options['batch_size'], height, width, num_channels))
		targets = np.zeros((inputs.shape[0], options['num_actions']))
		for i in range(len(data_batch)):
			x_now, action_idx, reward, x_next, is_dead, _ = data_batch[i]
			inputs[i: i+1] = x_now
			targets[i] = self.model.predict(x_now)
			Q_next = self.model.predict(x_next)
			if is_dead:
				targets[i, action_idx] = reward
			else:
				targets[i, action_idx] = reward + options['rd_gamma'] * np.max(Q_next)
		loss = self.model.train_on_batch(inputs, targets)
		return loss
	# scrore转reward
	def score2reward(self, score, is_dead, actions):
		'''
		reward = 0.12
		if actions[1] > actions[0]:
			reward = 0.1
		if is_dead:
			reward = -100
		'''
		reward = 0.1
		if is_dead:
			reward = -1
		return reward
	# 导入权重
	def load_weight(self, weight_path):
		self.model.load_weights(weight_path)
	# 保存模型信息
	def save_model_info(self, savename='model.json'):
		with open(savename, 'w') as f:
			json.dump(self.model.to_json(), f)
	def __repr__(self):
		return '[Model]:\n%s' % self.model
	def __str__(self):
		return '[INFO]: model_1-CNN built by keras...'


'''
Function:
	模型二: Torch框架搭建
'''
'''
class model_2(nn.Module):
	def __init__(self, options):
		super(model_2, self).__init__()
		# in_channels, out_channels, kernel_size, stride
		self.conv1 = nn.Conv2d(6, 32, 7, 2, padding=3)
		self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1)
		self.conv3 = nn.Conv2d(64, 64, 3, 2, padding=1)
		self.fc1 = nn.Linear(64 * 4 * 8, 256)
		self.fc2 = nn.Linear(256, options['num_actions'])
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc1(x.view(x.size(0), -1)))
		x = self.fc2(x)
		return x
'''

'''
Function:
	模型三: TensorFlow框架搭建
'''