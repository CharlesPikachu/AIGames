'''
Function:
	config file.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os


'''train'''
batch_size = 32
max_explore_iterations = 5000
max_memory_size = 50000
max_train_iterations = 1000000
save_interval = 10000
save_dir = 'model_saved'
num_continuous_frames = 6
image_size = (224, 224)
logfile = 'train.log'
use_cuda = True
eps_start = 1.0 # prob to explore at first
eps_end = 0.1 # prob to explore finally
eps_num_steps = 10000

'''test'''
weightspath = os.path.join(save_dir, str(max_train_iterations)+'.pkl') # trained model path

'''game'''
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
SKYBLUE = (0, 191, 255)
layout_filepath = 'layouts/mediumClassic.lay' # decide the game map
ghost_image_paths = [(each.split('.')[0], os.path.join(os.getcwd(), each)) for each in ['gameAPI/images/Blinky.png', 'gameAPI/images/Clyde.png', 'gameAPI/images/Inky.png', 'gameAPI/images/Pinky.png']]
scaredghost_image_path = os.path.join(os.getcwd(), 'gameAPI/images/scared.png')
pacman_image_path = ('pacman', os.path.join(os.getcwd(), 'gameAPI/images/pacman.png'))
font_path = os.path.join(os.getcwd(), 'gameAPI/font/ALGER.TTF')
grid_size = 32
operator = 'ai' # 'person' or 'ai', used in demo.py
ghost_action_method = 'catchup' # 'random' or 'catchup', ghost using 'catchup' is more intelligent than 'random'.