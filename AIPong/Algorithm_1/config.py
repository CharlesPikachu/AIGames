'''
Function:
	config file.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''


mode = 'train' # 'train' or 'test'
lr = 2e-4
batch_size = 32
model_dir = 'modelSaved'
logfile = 'train.log'
init_prob = 1.0
end_prob = 0.01
prob_num_steps = 10000
frame_size = (96, 96)
max_memory_size = 100000
num_explore_steps = 5000
save_interval = 10000
max_train_iteration = 1000000