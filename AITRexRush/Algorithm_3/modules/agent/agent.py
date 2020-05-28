'''
Function:
	define the game agent
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
from modules.agent.ga import *
from modules.sprites.dinosaur import *


'''define the agent'''
class Agent():
	def __init__(self, cfg, sounds, **kwargs):
		self.cfg = cfg
		self.sounds = sounds
		self.population_size = kwargs.get('population_size', 100)
		self.checkpointspath = kwargs.get('checkpointspath', 'checkpoints')
		if not os.path.exists(self.checkpointspath):
			os.mkdir(self.checkpointspath)
		self.checkpointspath = os.path.join(self.checkpointspath, 'ga.pkl')
		self.ai = GeneticModel(population_size=self.population_size)
		self.dinos = [Dinosaur(cfg.IMAGE_PATHS['dino']) for _ in range(self.population_size)]
		self.num_iter = 0
	'''update all dinos'''
	def update(self):
		for dino in self.dinos:
			if dino.is_dead: continue
			dino.update()
	'''draw on the screen'''
	def draw(self, screen):
		for dino in self.dinos:
			if dino.is_dead: continue
			dino.draw(screen)
	'''make decision for all dinos'''
	def makedecision(self, x):
		threshold = 0.55
		actions = self.ai.predict(x)
		for i in range(len(actions)):
			action = actions[i]
			if self.dinos[i].is_dead:
				continue
			if action[0] >= threshold:
				self.dinos[i].jump(self.sounds)
			elif action[1] >= threshold:
				self.dinos[i].duck()
			else:
				self.dinos[i].unduck()
			self.ai.populations[i].fitness = self.dinos[i].score
	'''next generation'''
	def nextgeneration(self):
		self.num_iter += 1
		self.dinos = [Dinosaur(self.cfg.IMAGE_PATHS['dino']) for _ in range(self.population_size)]
		self.ai.nextgeneration()
	'''save the model'''
	def save(self):
		self.ai.keepbest()
		self.ai.save(self.checkpointspath)
	'''load the model'''
	def load(self):
		self.ai.load(self.checkpointspath)