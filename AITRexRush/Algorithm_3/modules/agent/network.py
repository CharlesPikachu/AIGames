'''
Function:
	define the network
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import numpy as np


'''define the network'''
class Network():
	def __init__(self, fc1=None, fc2=None, **kwargs):
		self.fc1 = np.random.randn(5, 16) if fc1 is None else fc1
		self.fc2 = np.random.randn(16, 2) if fc2 is None else fc2
		self.fitness = 0
	'''predict the action'''
	def predict(self, x):
		x = x.dot(self.fc1)
		x = self.activation(x)
		x = x.dot(self.fc2)
		x = self.activation(x)
		return x
	'''activation function'''
	def activation(self, x):
		return 0.5 * (1 + np.tanh(0.5 * x))