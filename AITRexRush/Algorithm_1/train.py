# 作者: Charles
# 公众号: Charles的皮卡丘
# 训练函数
from nets.nets import *
from utils.api import *
from utils.utils import *


# for keras
# options = {
# 				"lr": 1e-4,
# 				"batch_size": 32,
# 				"max_batch": 150000,
# 				"input_shape": (96, 96, 4),
# 				"num_actions": 2,
# 				"savename": "autodino",
# 				"savepath": "./model",
# 				"log_dir": "logger",
# 				"data_dir": "data",
# 				"start_prob_jump": 0,
# 				"end_prob_jump": 0,
# 				"interval_prob": 1e5,
# 				"save_interval": 500,
# 				"num_operations": 200,
# 				"data_memory": 50000,
# 				"rd_gamma": 0.99,
# 				"use_canny": False,
# 				"num_sampling": 200,
# 				"game_url": 'chrome://dino'
# 			}
options = {
				"lr": 1e-4,
				"batch_size": 32,
				"max_batch": 150000,
				"input_shape": (96, 96, 4),
				"num_actions": 2,
				"savename": "autodino",
				"savepath": "./model",
				"log_dir": "logger",
				"data_dir": "data",
				"start_prob_jump": 0.1,
				"end_prob_jump": 1e-4,
				"interval_prob": 1e5,
				"save_interval": 1500,
				"num_operations": 200,
				"data_memory": 50000,
				"rd_gamma": 0.99,
				"use_canny": False,
				"num_sampling": 200,
				"game_url": 'chrome://dino'
			}
# for torch
'''
options = {
				
			}
'''


# 训练函数
def train():
	agent = DinoAgent(options['game_url'])
	# model = model_1(options)
	model = model_1(options, with_pool=True)
	# 如果你需要继续训练模型，请去除下面的注释并指定正确的模型权重路径
	model.load_weight('./model/autodino_100000.h5')
	model.train(agent, options, batch_idx=100000)



if __name__ == '__main__':
	train()