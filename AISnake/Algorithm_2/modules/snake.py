'''
Function:
	define the snake
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import copy
import random
import pygame


'''snake'''
class Snake(pygame.sprite.Sprite):
	def __init__(self, cfg, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.cfg = cfg
		self.head_coord = (random.randint(5, cfg.GAME_MATRIX_SIZE[0]-6), random.randint(5, cfg.GAME_MATRIX_SIZE[1]-6))
		self.tail_coords = []
		for i in range(1, 3):
			self.tail_coords.append((self.head_coord[0]-i, self.head_coord[1]))
		self.direction = 'right'
		self.head_colors = [(0, 80, 255), (0, 255, 255)]
		self.tail_colors = [(0, 155, 0), (0, 255, 0)]
	'''set the direction'''
	def setDirection(self, direction):
		assert direction in ['up', 'down', 'right', 'left']
		if direction == 'up':
			if self.head_coord[1]-1 != self.tail_coords[0][1]:
				self.direction = direction
		elif direction == 'down':
			if self.head_coord[1]+1 != self.tail_coords[0][1]:
				self.direction = direction
		elif direction == 'left':
			if self.head_coord[0]-1 != self.tail_coords[0][0]:
				self.direction = direction
		elif direction == 'right':
			if self.head_coord[0]+1 != self.tail_coords[0][0]:
				self.direction = direction
	'''update the snake'''
	def update(self, apple):
		# move according to the specified direction
		self.tail_coords.insert(0, copy.deepcopy(self.head_coord))
		if self.direction == 'up':
			self.head_coord = self.head_coord[0], self.head_coord[1] - 1
		elif self.direction == 'down':
			self.head_coord = self.head_coord[0], self.head_coord[1] + 1
		elif self.direction == 'left':
			self.head_coord = self.head_coord[0] - 1, self.head_coord[1]
		elif self.direction == 'right':
			self.head_coord = self.head_coord[0] + 1, self.head_coord[1]
		# whether snake has eaten the food
		if self.head_coord == apple.coord:
			return True
		else:
			self.tail_coords = self.tail_coords[:-1]
			return False
	'''draw it on the screen'''
	def draw(self, screen):
		head_x, head_y = self.head_coord[0] * self.cfg.BLOCK_SIZE, self.head_coord[1] * self.cfg.BLOCK_SIZE
		rect = pygame.Rect(head_x, head_y, self.cfg.BLOCK_SIZE, self.cfg.BLOCK_SIZE)
		pygame.draw.rect(screen, self.head_colors[0], rect)
		rect = pygame.Rect(head_x+4, head_y+4, self.cfg.BLOCK_SIZE-8, self.cfg.BLOCK_SIZE-8)
		pygame.draw.rect(screen, self.head_colors[1], rect)
		for coord in self.tail_coords:
			x, y = coord[0] * self.cfg.BLOCK_SIZE, coord[1] * self.cfg.BLOCK_SIZE
			rect = pygame.Rect(x, y, self.cfg.BLOCK_SIZE, self.cfg.BLOCK_SIZE)
			pygame.draw.rect(screen, self.tail_colors[0], rect)
			rect = pygame.Rect(x+4, y+4, self.cfg.BLOCK_SIZE-8, self.cfg.BLOCK_SIZE-8)
			pygame.draw.rect(screen, self.tail_colors[1], rect)
	'''get the snake matrix'''
	@property
	def coords(self):
		return [self.head_coord] + self.tail_coords
	'''whether the snake dies'''
	@property
	def isgameover(self):
		if (self.head_coord[0] < 0) or (self.head_coord[1] < 0) or \
		   (self.head_coord[0] >= self.cfg.GAME_MATRIX_SIZE[0]) or \
		   (self.head_coord[1] >= self.cfg.GAME_MATRIX_SIZE[1]):
			return True
		if self.head_coord in self.tail_coords:
			return True
		return False