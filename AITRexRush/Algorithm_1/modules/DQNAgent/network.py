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
		self.conv1 = nn.Conv2d(in_channels=num_input_frames, out_channels=32, kernel_size=9, stride=4, padding=4)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
		in_features = 64 * imagesize[0] * imagesize[1] // 64
		self.fc1 = nn.Linear(in_features=in_features, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
	def forward(self, x):
		batch_size = x.size(0)
		x = F.relu(self.conv1(x), inplace=True)
		x = F.relu(self.conv2(x), inplace=True)
		x = F.relu(self.conv3(x), inplace=True)
		x = x.view(batch_size, -1)
		x = F.relu(self.fc1(x), inplace=True)
		x = self.fc2(x)
		return x