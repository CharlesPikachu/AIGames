'''
Function:
	define the agent to play TRexRush
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import numpy as np


'''agent'''
class Agent():
	def __init__(self, bbox_area, **kwargs):
		self.bbox_area = bbox_area
		self.bg_color = 255
		self.reference_frame = np.full((bbox_area[3]-bbox_area[1], bbox_area[2]-bbox_area[0], 3), self.bg_color)
	'''return action according to the game frame'''
	def act(self, frame):
		action = [1, 0]
		frame = np.array(frame)
		if self.bg_color != frame[0][0][0]:
			self.bg_color = frame[0][0][0]
			self.reference_frame = np.full((self.bbox_area[3]-self.bbox_area[1], self.bbox_area[2]-self.bbox_area[0], 3), self.bg_color)
		diff = np.subtract(self.reference_frame, frame).sum()
		if diff != 0:
			action = [0, 1]
		return action