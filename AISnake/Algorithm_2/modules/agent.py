'''
Function:
	define the ai agent
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
from modules.food import *
from operator import itemgetter
from collections import OrderedDict


'''ai agent'''
class Agent():
	def __init__(self, cfg, snake, **kwargs):
		self.cfg = cfg
		self.num_rows = cfg.GAME_MATRIX_SIZE[1]
		self.num_cols = cfg.GAME_MATRIX_SIZE[0]
		self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
		self.path = self.buildcircle(snake)
		self.shortcut_path = {}
	'''make decision'''
	def act(self, snake, food):
		# make decision
		if self.shortcut_path:
			head_next = self.shortcut_path.pop(snake.coords[0])
		else:
			head_next = self.path[snake.coords[0]]
		query = (head_next[0]-snake.coords[0][0], head_next[1]-snake.coords[0][1])
		direction = {(-1, 0): 'left', (1, 0): 'right', (0, -1): 'up', (0, 1): 'down'}[query]
		snake.setDirection(direction)
		if snake.update(food):
			food = Apple(self.cfg, snake.coords)
			infos = {'eaten': True, 'food': food}
		else:
			infos = {'eaten': False, 'food': None}
		# if snake has eaten the food
		if head_next == food.coord:
			path = self.buildcircle(snake)
			if path:
				self.path = path
		# take shortcut
		if self.shortcut_path:
			return
		shortcut_path = self.shortcut(snake, food)
		if shortcut_path:
			self.shortcut_path = shortcut_path
		# return the necessary infos
		return infos
	'''calculate shortcut path'''
	def shortcut(self, snake, food):
		# empty screen, with the ordered hamitonian cycle precomputed and order numbered
		world = [[0 for i in range(self.num_cols)] for j in range(self.num_rows)]
		num = 1
		node = snake.coords[-1]
		world[node[1]][node[0]] = num
		node = self.path[node]
		while node != snake.coords[-1]:
			num += 1
			world[node[1]][node[0]] = num
			node = self.path[node]
		# obtain shortcut_path
		wall = snake.coords
		food = food.coord
		food_number = world[food[1]][food[0]]
		node, pre = wall[0], (-1, -1)
		wait = OrderedDict()
		wait[node] = pre
		path = {}
		while wait:
			node, pre = wait.popitem(last=False)
			path[node] = pre
			if node == food:
				break
			node_number = world[node[1]][node[0]]
			neigh = {}
			for direction in self.directions:
				to = (node[0]+direction[0], node[1]+direction[1])
				if not self.checkboundary(to):
					continue
				if to in wait or to in wall or to in path:
					continue
				to_number = world[to[1]][to[0]]
				if to_number > node_number and to_number <= food_number:
					neigh[node_number] = to
			neigh = sorted(neigh.items(), key=itemgetter(0), reverse=True)
			for item in neigh:
				wait[item[1]] = node
		if node != food:
			return {}
		return self.reverse(path, snake.coords[0], food)
	'''check boundary'''
	def checkboundary(self, pos):
		if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.num_cols or pos[1] >= self.num_rows:
			return False
		return True
	'''the shortest'''
	def shortest(self, wall, head, food):
		wait = OrderedDict()
		node, pre = head, (-1, -1)
		wait[node] = pre
		path = {}
		while wait:
			node, pre = wait.popitem(last=False)
			path[node] = pre
			if node == food:
				break
			if pre in path:
				prepre = path[pre]
				direction = (pre[0]-prepre[0], pre[1]-prepre[1])
				if (direction in self.directions) and (direction != self.directions[0]):
					self.directions.remove(direction)
					self.directions.insert(0, direction)
			for direction in self.directions:
				to = (node[0] + direction[0], node[1] + direction[1])
				if not self.checkboundary(to):
					continue
				if to in path or to in wait or to in wall:
					continue
				wait[to] = node
		if node != food:
			return None
		return self.reverse(path, head, food)
	'''reverse path'''
	def reverse(self, path, head, food):
		if not path: return path
		path_new = {}
		node = food
		while node != head:
			path_new[path[node]] = node
			node = path[node]
		return path_new
	'''the longest'''
	def longest(self, wall, head, food):
		path = self.shortest(wall, head, food)
		if path is None:
			return None
		node = head
		while node != food:
			if self.extendpath(path, node, wall+[food]):
				node = head
				continue
			node = path[node]
		return path
	'''extend path'''
	def extendpath(self, path, node, wall):
		next_ = path[node]
		direction_1 = (next_[0]-node[0], next_[1]-node[1])
		if direction_1 in [(0, -1), (0, 1)]:
			directions = [(-1, 0), (1, 0)]
		else:
			directions = [(0, -1), (0, 1)]
		for d in directions:
			src = (node[0]+d[0], node[1]+d[1])
			to = (next_[0]+d[0], next_[1]+d[1])
			if (src == to) or not (self.checkboundary(src) and self.checkboundary(to)):
				continue
			if src in path or src in wall or to in path or to in wall:
				continue
			direction_2 = (to[0]-src[0], to[1]-src[1])
			if direction_1 == direction_2:
				path[node] = src
				path[src] = to
				path[to] = next_
				return True
		return False
	'''build a Hamiltonian cycle'''
	def buildcircle(self, snake):
		path = self.longest(snake.coords[1: -1], snake.coords[0], snake.coords[-1])
		if (not path) or (len(path) - 1 != self.num_rows * self.num_cols - len(snake.coords)):
			return None
		for i in range(1, len(snake.coords)):
			path[snake.coords[i]] = snake.coords[i-1]
		return path