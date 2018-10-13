'''
工具函数
作者: Charles
公众号: Charles的皮卡丘
'''
import random
import numpy as np


# 选择行动(随机选择/有DQN选择)
def get_action_idx(q_values, prob, num_frame, OBSERVE, num_action=3):
	if (np.random.rand() <= prob) or (num_frame <= OBSERVE):
		action_idx = np.random.choice(num_action)
	else:
		action_idx = np.argmax(q_values)
	return action_idx


# 减小prob, 逐渐让神经网络来决定之后的行动
def down_prob(prob, num_frame, OBSERVE, EXPLORE, init_prob, end_prob):
	if (prob > end_prob) and (num_frame > OBSERVE):
		prob = prob - (init_prob - end_prob) / EXPLORE
	return prob