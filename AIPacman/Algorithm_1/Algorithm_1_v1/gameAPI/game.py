'''
Function:
	define the game agent
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import sys
import random
import pygame
import numpy as np
from .sprites import *


'''layout file parser'''
class LayoutParser():
	def __init__(self, config, **kwargs):
		self.gamemap = self.__parse(config.layout_filepath)
		self.height = len(self.gamemap)
		self.width = len(self.gamemap[0])
		config.frame_size = (self.height, self.width)
	'''parse .lay'''
	def __parse(self, filepath):
		gamemap = []
		f = open(filepath)
		for line in f.readlines():
			elements = []
			for c in line:
				if c == '%':
					elements.append('wall')
				elif c == '.':
					elements.append('food')
				elif c == 'o':
					elements.append('capsule')
				elif c == 'P':
					elements.append('Pacman')
				elif c in ['G']:
					elements.append('Ghost')
				elif c == ' ':
					elements.append(' ')
			gamemap.append(elements)
		f.close()
		return gamemap


'''define the game agent'''
class GamePacmanAgent():
	def __init__(self, config, **kwargs):
		self.config = config
		self.layout = LayoutParser(config)
		self.screen_width = self.layout.width * config.grid_size
		self.screen_height = self.layout.height * config.grid_size
		self.reset()
	'''next frame'''
	def nextFrame(self, action=None):
		if action is None:
			action = random.choice(self.actions)
		pygame.event.pump()
		pressed_keys = pygame.key.get_pressed()
		if pressed_keys[pygame.K_q]:
			sys.exit(-1)
			pygame.quit()
		is_win = False
		is_gameover = False
		reward = 0
		self.pacman_sprites.update(action, self.wall_sprites, None)
		for pacman in self.pacman_sprites:
			food_eaten = pygame.sprite.spritecollide(pacman, self.food_sprites, True)
			capsule_eaten = pygame.sprite.spritecollide(pacman, self.capsule_sprites, True)
		nonscared_ghost_sprites = pygame.sprite.Group()
		dead_ghost_sprites = pygame.sprite.Group()
		for ghost in self.ghost_sprites:
			if ghost.is_scared:
				if pygame.sprite.spritecollide(ghost, self.pacman_sprites, False):
					reward += 6
					dead_ghost_sprites.add(ghost)
			else:
				nonscared_ghost_sprites.add(ghost)
		for ghost in dead_ghost_sprites:
			ghost.reset()
		del dead_ghost_sprites
		reward += len(food_eaten) * 2
		reward += len(capsule_eaten) * 3
		if len(capsule_eaten) > 0:
			for ghost in self.ghost_sprites:
				ghost.is_scared = True
		self.ghost_sprites.update(self.wall_sprites, None, self.config.ghost_action_method, self.pacman_sprites)
		self.screen.fill(self.config.BLACK)
		self.wall_sprites.draw(self.screen)
		self.food_sprites.draw(self.screen)
		self.capsule_sprites.draw(self.screen)
		self.pacman_sprites.draw(self.screen)
		self.ghost_sprites.draw(self.screen)
		# get frame
		num_element_types = self.config.num_element_types
		frame = np.zeros((*self.config.frame_size, num_element_types))
		ghost_matrix = np.zeros(self.config.frame_size)
		scared_ghost_matrix = np.zeros(self.config.frame_size)
		food_matrix = np.zeros(self.config.frame_size)
		wall_matrix = np.zeros(self.config.frame_size)
		capsule_matrix = np.zeros(self.config.frame_size)
		pacman_matrix = np.zeros(self.config.frame_size)
		for ghost in self.ghost_sprites:
			y, x = ghost.rect.left // self.config.grid_size, ghost.rect.top // self.config.grid_size
			if ghost.is_scared:
				scared_ghost_matrix[x, y] = 1
			else:
				ghost_matrix[x, y] = 1
		for food in self.food_sprites:
			y, x = food.rect.left // self.config.grid_size, food.rect.top // self.config.grid_size
			food_matrix[x, y] = 1
		for wall in self.wall_sprites:
			y, x = wall.rect.left // self.config.grid_size, wall.rect.top // self.config.grid_size
			wall_matrix[x, y] = 1
		for capsule in self.capsule_sprites:
			y, x = capsule.rect.left // self.config.grid_size, capsule.rect.top // self.config.grid_size
			capsule_matrix[x, y] = 1
		for pacman in self.pacman_sprites:
			y, x = pacman.rect.left // self.config.grid_size, pacman.rect.top // self.config.grid_size
			pacman_matrix[x, y] = 1
		frame[:, :, 0] = wall_matrix
		frame[:, :, 1] = pacman_matrix
		frame[:, :, 2] = ghost_matrix
		frame[:, :, 3] = scared_ghost_matrix
		frame[:, :, 4] = food_matrix
		frame[:, :, 5] = capsule_matrix
		# show the score
		self.score += reward
		text = self.font.render('SCORE: %s' % self.score, True, self.config.WHITE)
		self.screen.blit(text, (2, 2))
		pygame.display.update()
		# judge whether game over
		if len(self.food_sprites) == 0 and len(self.capsule_sprites) == 0:
			is_win = True
			is_gameover = True
			reward = 10
		if pygame.sprite.groupcollide(self.pacman_sprites, nonscared_ghost_sprites, False, False):
			is_win = False
			is_gameover = True
			reward = -15
		if reward == 0:
			reward = -2
		return frame, is_win, is_gameover, reward, action
	'''run game(user control, for test)'''
	def runGame(self):
		clock = pygame.time.Clock()
		is_win = False
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit(-1)
					pygame.quit()
			pressed_keys = pygame.key.get_pressed()
			if pressed_keys[pygame.K_UP]:
				self.pacman_sprites.update([0, -1], self.wall_sprites, None)
			elif pressed_keys[pygame.K_DOWN]:
				self.pacman_sprites.update([0, 1], self.wall_sprites, None)
			elif pressed_keys[pygame.K_LEFT]:
				self.pacman_sprites.update([-1, 0], self.wall_sprites, None)
			elif pressed_keys[pygame.K_RIGHT]:
				self.pacman_sprites.update([1, 0], self.wall_sprites, None)
			for pacman in self.pacman_sprites:
				food_eaten = pygame.sprite.spritecollide(pacman, self.food_sprites, True)
				capsule_eaten = pygame.sprite.spritecollide(pacman, self.capsule_sprites, True)
			nonscared_ghost_sprites = pygame.sprite.Group()
			dead_ghost_sprites = pygame.sprite.Group()
			for ghost in self.ghost_sprites:
				if ghost.is_scared:
					if pygame.sprite.spritecollide(ghost, self.pacman_sprites, False):
						self.score += 6
						dead_ghost_sprites.add(ghost)
				else:
					nonscared_ghost_sprites.add(ghost)
			for ghost in dead_ghost_sprites:
				ghost.reset()
			self.score += len(food_eaten) * 2
			self.score += len(capsule_eaten) * 3
			if len(capsule_eaten) > 0:
				for ghost in self.ghost_sprites:
					ghost.is_scared = True
			self.ghost_sprites.update(self.wall_sprites, None, self.config.ghost_action_method, self.pacman_sprites)
			self.screen.fill(self.config.BLACK)
			self.wall_sprites.draw(self.screen)
			self.food_sprites.draw(self.screen)
			self.capsule_sprites.draw(self.screen)
			self.pacman_sprites.draw(self.screen)
			self.ghost_sprites.draw(self.screen)
			# show the score
			text = self.font.render('SCORE: %s' % self.score, True, self.config.WHITE)
			self.screen.blit(text, (2, 2))
			# judge whether game over
			if len(self.food_sprites) == 0 and len(self.capsule_sprites) == 0:
				is_win = True
				break
			if pygame.sprite.groupcollide(self.pacman_sprites, nonscared_ghost_sprites, False, False):
				is_win = False
				break
			pygame.display.flip()
			clock.tick(10)
		if is_win:
			self.__showText(msg='You won!', position=(self.screen_width//2-50, int(self.screen_height/2.5)))
		else:
			self.__showText(msg='Game Over!', position=(self.screen_width//2-80, int(self.screen_height/2.5)))
	'''reset'''
	def reset(self):
		self.screen, self.font = self.__initScreen()
		self.wall_sprites, self.pacman_sprites, self.ghost_sprites, self.capsule_sprites, self.food_sprites = self.__createGameMap()
		self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
		self.score = 0
	'''show the game info'''
	def __showText(self, msg, position):
		clock = pygame.time.Clock()
		text = self.font.render(msg, True, self.config.WHITE)
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()
					pygame.quit()
			self.screen.fill(self.config.BLACK)
			self.screen.blit(text, position)
			pygame.display.flip()
			clock.tick(10)
	def __createGameMap(self):
		wall_sprites = pygame.sprite.Group()
		pacman_sprites = pygame.sprite.Group()
		ghost_sprites = pygame.sprite.Group()
		capsule_sprites = pygame.sprite.Group()
		food_sprites = pygame.sprite.Group()
		ghost_idx = 0
		for i in range(self.layout.height):
			for j in range(self.layout.width):
				elem = self.layout.gamemap[i][j]
				if elem == 'wall':
					position = [j*self.config.grid_size, i*self.config.grid_size]
					wall_sprites.add(Wall(*position, self.config.grid_size, self.config.grid_size, self.config.SKYBLUE))
				elif elem == 'food':
					position = [j*self.config.grid_size+self.config.grid_size*0.5, i*self.config.grid_size+self.config.grid_size*0.5]
					food_sprites.add(Food(*position, 10, 10, self.config.GREEN, self.config.WHITE))
				elif elem == 'capsule':
					position = [j*self.config.grid_size+self.config.grid_size*0.5, i*self.config.grid_size+self.config.grid_size*0.5]
					capsule_sprites.add(Food(*position, 16, 16, self.config.GREEN, self.config.WHITE))
				elif elem == 'Pacman':
					position = [j*self.config.grid_size+self.config.grid_size*0.5, i*self.config.grid_size+self.config.grid_size*0.5]
					pacman_sprites.add(Pacman(*position, self.config.pacman_image_path, (self.config.grid_size, self.config.grid_size)))
				elif elem == 'Ghost':
					position = [j*self.config.grid_size+self.config.grid_size*0.5, i*self.config.grid_size+self.config.grid_size*0.5]
					ghost_sprites.add(Ghost(*position, self.config.ghost_image_paths[ghost_idx], self.config.scaredghost_image_path, (self.config.grid_size, self.config.grid_size)))
					ghost_idx += 1
		return wall_sprites, pacman_sprites, ghost_sprites, capsule_sprites, food_sprites
	'''initialize the game screen'''
	def __initScreen(self):
		pygame.init()
		pygame.font.init()
		screen = pygame.display.set_mode([self.screen_width, self.screen_height])
		pygame.display.set_caption('Pacman-微信公众号:Charles的皮卡丘')
		font = pygame.font.Font(self.config.font_path, 24)
		return screen, font