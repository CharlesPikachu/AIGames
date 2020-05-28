'''
Function:
	define the genetic algorithm model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import copy
import random
import pickle
import numpy as np
from modules.agent.network import *


'''genetic algorithm model'''
class GeneticModel():
	def __init__(self, **kwargs):
		self.population_size = kwargs.get('population_size', 100)
		self.num_keeped_nets = kwargs.get('num_keeped_nets', 2)
		self.num_crossover_times = kwargs.get('num_crossover_times', 2)
		self.mutation_prob = kwargs.get('mutation_prob', 0.5)
		self.populations = [Network() for _ in range(self.population_size)]
		self.keeped_nets = []
	'''predict the next action for dinos'''
	def predict(self, x):
		x = np.array(x)
		preds = [net.predict(x) for net in self.populations]
		return preds
	'''keep the best network so far'''
	def keepbest(self):
		self.populations.sort(key=lambda x: x.fitness, reverse=True)
		self.keeped_nets = self.populations[:self.num_keeped_nets]
	'''obtain next generation'''
	def nextgeneration(self):
		self.keepbest()
		self.populations = copy.deepcopy(self.keeped_nets)
		nets_crossover = self.crossover()
		for item in nets_crossover:
			self.populations.append(self.mutate(item))
		nets_new = []
		size = self.population_size - len(self.populations)
		if size > 0:
			for i in range(size):
				net = copy.deepcopy(random.choice(self.populations))
				nets_new.append(self.mutate(net))
		self.populations += nets_new
		if len(self.populations) > self.population_size:
			self.populations = self.populations[:self.population_size]
	'''crossover'''
	def crossover(self):
		def crossoverweight(fc1, fc2):
			assert len(fc1) == len(fc2)
			crossover_len = int(len(fc1) * random.uniform(0, 1))
			for i in range(crossover_len):
				fc1[i], fc2[i] = fc2[i], fc1[i]
			return fc1, fc2
		nets_new = []
		size = min(self.num_keeped_nets * self.num_keeped_nets, self.population_size)
		for _ in range(size):
			net_1 = copy.deepcopy(random.choice(self.keeped_nets))
			net_2 = copy.deepcopy(random.choice(self.keeped_nets))
			for _ in range(self.num_crossover_times):
				net_1.fc1, net_2.fc1 = crossoverweight(net_1.fc1, net_2.fc1)
				net_1.fc2, net_2.fc2 = crossoverweight(net_1.fc2, net_2.fc2)
				nets_new.append(net_1)
		return nets_new
	'''mutate'''
	def mutate(self, net):
		def mutateweight(fc, prob):
			if random.uniform(0, 1) < prob:
				return fc * random.uniform(0.5, 1.5)
			else:
				return fc
		net = copy.deepcopy(net)
		net.fc1 = mutateweight(net.fc1, self.mutation_prob)
		net.fc2 = mutateweight(net.fc2, self.mutation_prob)
		return net
	'''save the model'''
	def save(self, filepath):
		fp = open(filepath, 'wb')
		pickle.dump(self.keeped_nets, fp)
		fp.close()
	'''load the model'''
	def load(self, filepath):
		if not os.path.isfile(filepath):
			return
		fp = open(filepath, 'rb')
		self.keeped_nets = pickle.load(filepath)
		self.populations = copy.deepcopy(self.keeped_nets)
		for i in range(self.population_size - len(self.keeped_nets)):
			self.populations.append(Network())