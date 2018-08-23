# 作者: Charles
# 公众号: Charles的皮卡丘
# 调用训练好的模型自动玩T-Rex Rush
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
# 				"start_prob_jump": 0.1,
# 				"end_prob_jump": 1e-4,
# 				"interval_prob": 1e5,
# 				"save_interval": 1500,
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
				"start_prob_jump": 0,
				"end_prob_jump": 0,
				"interval_prob": 1e5,
				"save_interval": 500,
				"num_operations": 200,
				"data_memory": 50000,
				"rd_gamma": 0.99,
				"use_canny": False,
				"num_sampling": 200,
				"game_url": 'chrome://dino'
			}


'''
Function:
	自动玩T-Rex Rush
Input:
	-model: 模型
	-model_type: 模型搭建的框架
'''
def auto_play(model, model_type, trained_model):
	if model_type == 'keras':
		agent = DinoAgent(options['game_url'])
		model = model_1(options, with_pool=True)
		model.load_weight('./model/autodino_100000.h5')
		model.train(agent, options)
	elif model_type == 'torch':
		pass
	elif model_type == 'tensorflow':
		pass
	elif model_type == 'mxnet':
		pass
	elif model_type == 'caffe':
		pass
	elif model_type == 'theano':
		pass
	else:
		return None





if __name__ == '__main__':
	model = model_1(options, with_pool=True)
	auto_play(model, model_type='keras', trained_model='./model/autodino_87000.h5')