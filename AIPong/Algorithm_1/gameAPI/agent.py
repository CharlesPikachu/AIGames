'''
Function:
	game agent
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import random
from .PongGame import PongGame


'''game agent'''
class gameAgent():
	def __init__(self):
		self.game = PongGame()
		self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	'''next frame'''
	def nextFrame(self, action=None):
		if action is None:
			action = random.choice(self.actions)
		return self.game.nextFrame(action)