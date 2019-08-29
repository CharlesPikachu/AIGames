'''
Function:
	create the network.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import deque
from gameAPI.agent import gameAgent


'''Deep Q-network'''
class DQN():
	def __init__(self, config, **kwargs):
		self.config = config
	'''train the net'''
	def train(self, session):
		# create network
		net_in, net_out = self.__createNetwork()
		action_placeholder = tf.placeholder('float', [None, 3])
		target_placeholder = tf.placeholder('float', [None])
		loss = tf.reduce_mean(tf.square(target_placeholder - tf.reduce_sum(tf.multiply(net_out, action_placeholder), reduction_indices=1)))
		train_step = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
		saver = tf.train.Saver()
		session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state(self.config.model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(session, checkpoint.model_checkpoint_path)
			self.__logging('[INFO]: Load %s successfully...' % checkpoint.model_checkpoint_path, self.config.logfile)
		else:
			self.__logging('[INFO]: No checkpoint found, start to train a new model...', self.config.logfile)
		# game agent
		game_agent = gameAgent()
		# cache
		data_deque = deque()
		# record frames now
		num_frames = 0
		# record games now
		num_games = 0
		# num of win games
		num_win_games = 0
		# prob
		if self.config.mode == 'train':
			prob = self.config.init_prob
		elif self.config.mode == 'test':
			prob = self.config.end_prob
		else:
			raise ValueError('Hhhhh, config.mode is wrong, should be <train> or <test>...')
		# frame prev
		frame_pre = None
		frame_now = None
		# run game
		while True:
			# random decide
			if random.random() <= prob or frame_pre is None:
				frame_now, action, reward, terminal, paddle_1_score, paddle_2_score = game_agent.nextFrame(action=None)
				frame_now = cv2.resize(frame_now, self.config.frame_size)
			# decide by network
			else:
				action_pred = net_out.eval(feed_dict={net_in: [frame_pre]})
				action_pred = np.argmax(action_pred)
				if action_pred == 0:
					action_pred = [1, 0, 0]
				elif action_pred == 1:
					action_pred = [0, 1, 0]
				elif action_pred == 2:
					action_pred = [0, 0, 1]
				else:
					raise RuntimeError('Hhhhh, your code in net_out for action should be wrong I think...')
				frame_now, action, reward, terminal, paddle_1_score, paddle_2_score = game_agent.nextFrame(action=action_pred)
				frame_now = cv2.resize(frame_now, self.config.frame_size) / 255.
			# save data
			if frame_pre is not None:
				data_deque.append((frame_pre, frame_now, terminal, reward, np.array(action)))
			if len(data_deque) > self.config.max_memory_size:
				data_deque.popleft()
			frame_pre = frame_now.copy()
			# prob decrease
			if prob > self.config.end_prob:
				prob = max(self.config.end_prob, prob-(self.config.init_prob-self.config.end_prob)/self.config.num_explore_steps)
			# statistic
			num_frames += 1
			num_games += int(terminal)
			if terminal:
				num_win_games += int(paddle_1_score > paddle_2_score)
			# train
			if self.config.mode == 'train':
				# training
				if len(data_deque) > self.config.num_explore_steps:
					minibatch = random.sample(data_deque, self.config.batch_size)
					frame_pre_batch = [mb[0] for mb in minibatch]
					frame_now_batch = [mb[1] for mb in minibatch]
					terminal_batch = [mb[2] for mb in minibatch]
					reward_batch = [mb[3] for mb in minibatch]
					action_batch = [mb[4] for mb in minibatch]
					frame_now_out = net_out.eval(feed_dict={net_in: frame_now_batch})
					target_batch = []
					for i in range(self.config.batch_size):
						if terminal_batch[i]:
							target_batch.append(reward_batch[i])
						else:
							target_batch.append(reward_batch[i] + 0.95 * np.max(frame_now_out[i]))
					train_step.run(feed_dict={target_placeholder: target_batch,
											  action_placeholder: action_batch,
											  net_in: frame_pre_batch})
					if num_frames % self.config.save_interval == 0:
						saver.save(session, os.path.join(self.config.model_dir, 'DQN_Pong'), global_step=num_frames)
						self.__logging('[INFO]: Save model in %s...' % self.config.model_dir, self.config.logfile)
					log_content = '[STATE]: Train, [FRAMES]: %s, [GAMES]: %s, [WINS]: %s, [LOSS]: %s...' % (num_frames, num_games, num_win_games, str(session.run(loss, feed_dict={target_placeholder: target_batch,
																																												   action_placeholder: action_batch,
																																												   net_in: frame_pre_batch})))
				# observing
				else:
					log_content = '[STATE]: Oberve, [FRAMES]: %s, [GAMES]: %s, [WINS]: %s...' % (num_frames, num_games, num_win_games)
			# test
			elif self.config.mode == 'test':
				log_content = '[STATE]: Test, [FRAME]: %s, [GAME]: %s, [WINS]: %s...' % (num_frames, num_games, num_win_games)
			# parse error
			else:
				raise ValueError('Hhhhh, config.mode is wrong, should be <train> or <test>...')
			# print and save info
			self.__logging(log_content, self.config.logfile)
			# break game if greater than limitation
			if num_frames > self.config.max_iterations:
				break
	'''create network'''
	def __createNetwork(self):
		w_conv1 = self.__initWeightVariable([8, 8, 3, 32])
		b_conv1 = self.__initBiasVariable([32])
		w_conv2 = self.__initWeightVariable([4, 4, 32, 64])
		b_conv2 = self.__initBiasVariable([64])
		w_conv3 = self.__initWeightVariable([3, 3, 64, 64])
		b_conv3 = self.__initBiasVariable([64])
		w_fc1 = self.__initWeightVariable([(self.config.frame_size[0]//8)*(self.config.frame_size[1]//8)*64, 512])
		b_fc1 = self.__initBiasVariable([512])
		w_fc2 = self.__initWeightVariable([512, 3])
		b_fc2 = self.__initBiasVariable([3])
		x = tf.placeholder('float', [None, self.config.frame_size[0], self.config.frame_size[1], 3])
		conv1 = tf.nn.leaky_relu(self.__conv2d(x, w_conv1, 4)+b_conv1, alpha=0.1)
		conv2 = tf.nn.leaky_relu(self.__conv2d(conv1, w_conv2, 2)+b_conv2, alpha=0.1)
		conv3 = tf.nn.leaky_relu(self.__conv2d(conv2, w_conv3, 1)+b_conv3, alpha=0.1)
		flatten = tf.reshape(conv3, [-1, (self.config.frame_size[0]//8)*(self.config.frame_size[1]//8)*64])
		fc1 = tf.nn.leaky_relu(tf.matmul(flatten, w_fc1)+b_fc1, alpha=0.1)
		fc2 = tf.matmul(fc1, w_fc2) + b_fc2
		return x, fc2
	'''initial weight'''
	def __initWeightVariable(self, shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.01))
	'''initial bias'''
	def __initBiasVariable(self, shape):
		return tf.Variable(tf.constant(0.01, shape=shape))
	'''conv2d'''
	def __conv2d(self, x, weights, stride):
		return tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding="SAME")
	'''maxpool'''
	def __maxpool(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	'''logging'''
	def __logging(self, content, logfile):
		with open(logfile, 'a') as f:
			f.write(content + '\n')
		print(content)