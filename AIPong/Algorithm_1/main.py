'''
Function:
	train the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import config
import tensorflow as tf
from model.model import DQN


'''train'''
def train():
	session = tf.InteractiveSession()
	DQN(config).train(session)


'''run'''
if __name__ == '__main__':
	train()