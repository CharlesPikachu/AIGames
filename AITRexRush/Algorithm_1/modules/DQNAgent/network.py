'''
Function:
	define the network
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''build network'''
class DeepQNetwork(nn.Module):
	def __init__(self, imagesize, num_input_frames, num_actions, **kwargs):
		super(DeepQNetwork, self).__init__()
		assert imagesize == (84, 84)
		self.conv1 = nn.Conv2d(in_channels=num_input_frames, out_channels=32, kernel_size=8, stride=4, padding=0)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.fc1 = nn.Linear(in_features=3136, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
	def forward(self, x):
		batch_size = x.size(0)
		x = self.conv1(x)
		x = F.relu(x, inplace=True)
		x = self.conv2(x)
		x = F.relu(x, inplace=True)
		x = self.conv3(x)
		x = F.relu(x, inplace=True)
		x = x.view(batch_size, -1)
		x = F.relu(self.fc1(x), inplace=True)
		x = self.fc2(x)
		return x
	@staticmethod
	def initWeights(m):
		if type(m) == nn.Conv2d or type(m) == nn.Linear:
			nn.init.uniform_(m.weight, -0.01, 0.01)
			m.bias.data.fill_(0.01)