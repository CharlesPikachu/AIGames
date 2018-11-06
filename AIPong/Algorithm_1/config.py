'''
配置文件
作者: Charles
公众号: Charles的皮卡丘
'''
options = {
			'info': 'DQN for game of Pong',
			'num_action': 3,
			'lr': 1e-6,
			'batch_size': 32,
			'modelDir': 'modelSaved',
			'init_prob': 1.0,
			'end_prob': 1e-6,
			'OBSERVE': 50000,
			'EXPLORE': 50000,
			'REPLAY_MEMORY': 500000,
			'action_interval': 7,
			'gamma': 0.99,
			'save_interval': 20000,
			'logfile': 'train.log',
			'is_train': True
			}