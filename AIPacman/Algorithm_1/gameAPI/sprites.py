'''
Function:
	define the game sprites.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import pygame
import random


'''define the wall'''
class Wall(pygame.sprite.Sprite):
	def __init__(self, x, y, width, height, color, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.Surface([width, height])
		self.image.fill(color)
		self.rect = self.image.get_rect()
		self.rect.left = x
		self.rect.top = y


'''define the food'''
class Food(pygame.sprite.Sprite):
	def __init__(self, x, y, width, height, color, bg_color, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.Surface([width, height])
		self.image.fill(bg_color)
		self.image.set_colorkey(bg_color)
		pygame.draw.ellipse(self.image, color, [0, 0, width, height])
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)


'''define the ghost'''
class Ghost(pygame.sprite.Sprite):
	def __init__(self, x, y, role_image_path, scaredghost_image_path, image_size, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.ori_x, self.ori_y = x, y
		self.role_name = role_image_path[0]
		self.scared_image = pygame.image.load(scaredghost_image_path).convert()
		self.scared_image = pygame.transform.scale(self.scared_image, image_size)
		self.base_image = pygame.image.load(role_image_path[1]).convert()
		self.base_image = pygame.transform.scale(self.base_image, image_size)
		self.image = self.base_image.copy()
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)
		self.prev_x = x
		self.prev_y = y
		self.base_speed = [16, 16]
		self.speed = [0, 0]
		self.direction_now = None
		self.direction_legal = []
		self.directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
		self.is_scared = False
		self.is_scared_timer = 40
		self.is_scared_count = 0
		self.random_step_first = 50
	'''update'''
	def update(self, wall_sprites, gate_sprites, method='random', pacman_sprites=None):
		if self.random_step_first > 0:
			self.random_step_first -= 1
			method = 'random'
		if self.is_scared:
			self.base_speed = [8, 8]
			self.is_scared_count += 1
			if self.is_scared_count > self.is_scared_timer:
				self.is_scared_count = 0
				self.is_scared = False
		else:
			self.base_speed = [16, 16]
		self.direction_now = self.__randomChoice(self.__getLegalAction(wall_sprites, gate_sprites), method, pacman_sprites)
		# update attributes
		ori_image = self.base_image if not self.is_scared else self.scared_image
		if self.direction_now[0] < 0:
			self.image = pygame.transform.flip(ori_image, True, False)
		elif self.direction_now[0] > 0:
			self.image = ori_image.copy()
		elif self.direction_now[1] < 0:
			self.image = pygame.transform.rotate(ori_image, 90)
		elif self.direction_now[1] > 0:
			self.image = pygame.transform.rotate(ori_image, -90)
		self.speed = [self.direction_now[0] * self.base_speed[0], self.direction_now[1] * self.base_speed[1]]
		# move
		self.rect.left += self.speed[0]
		self.rect.top += self.speed[1]
		return True
	'''reset'''
	def reset(self):
		self.random_step_first = 50
		self.rect.center = (self.ori_x, self.ori_y)
		self.is_scared_count = 0
		self.is_scared = False
	'''get the legal action'''
	def __getLegalAction(self, wall_sprites, gate_sprites):
		direction_legal = []
		for direction in self.directions:
			if self.__isActionLegal(direction, wall_sprites, gate_sprites):
				direction_legal.append(direction)
		if sorted(direction_legal) == sorted(self.direction_legal):
			return [self.direction_now]
		else:
			self.direction_legal = direction_legal
			return self.direction_legal
	'''random choice the action'''
	def __randomChoice(self, directions, method='random', pacman_sprites=None):
		if method == 'random':
			return random.choice(directions)
		elif method == 'catchup':
			for pacman in pacman_sprites:
				pacman_pos = pacman.rect.center
			distances = []
			for direction in directions:
				speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
				ghost_pos = (self.rect.left+speed[0], self.rect.top+speed[1])
				distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
				distances.append([distance, direction])
			if self.is_scared:
				best_score = max([d[0] for d in distances])
				best_prob = 0.8
			else:
				best_score = min([d[0] for d in distances])
				best_prob = 0.8
			best_directions = [d[1] for d in distances if d[0] == best_score]
			probs = {}
			for each in directions:
				probs[self.__formatDirection(each)] = (1 - best_prob) / len(directions)
			for each in best_directions:
				probs[self.__formatDirection(each)] += best_prob / len(best_directions)
			total = float(sum(probs.values()))
			for key in list(probs.keys()):
				probs[key] = probs[key] / total
			r = random.random()
			base = 0.0
			for key, value in probs.items():
				base += value
				if r <= base:
					return self.__formatDirection(key)
		else:
			raise ValueError('Unsupport method %s in Ghost.__randomChoice...' % method)
	'''direction format change'''
	def __formatDirection(self, direction):
		if isinstance(direction, str):
			directions_dict = {'left': [-1, 0], 'right': [1, 0], 'up': [0, -1], 'down': [0, 1]}
			direction = directions_dict.get(direction)
			if direction is None:
				raise ValueError('Error value %s in Ghost.__formatDirection...' % str(direction))
			else:
				return direction
		elif isinstance(direction, list):
			if direction == [-1, 0]:
				return 'left'
			elif direction == [1, 0]:
				return 'right'
			elif direction == [0, -1]:
				return 'up'
			elif direction == [0, 1]:
				return 'down'
			else:
				raise ValueError('Error value %s in Ghost.__formatDirection...' % str(direction))
		else:
			raise ValueError('Unsupport direction format %s in Ghost.__formatDirection...' % type(direction))
	'''judge whether the action legal'''
	def __isActionLegal(self, direction, wall_sprites, gate_sprites):
		speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
		x_prev = self.rect.left
		y_prev = self.rect.top
		self.rect.left += speed[0]
		self.rect.top += speed[1]
		is_collide = pygame.sprite.spritecollide(self, wall_sprites, False)
		if gate_sprites is not None:
			if not is_collide:
				is_collide = pygame.sprite.spritecollide(self, gate_sprites, False)
		self.rect.left = x_prev
		self.rect.top = y_prev
		return not is_collide


'''define the Pacman'''
class Pacman(pygame.sprite.Sprite):
	def __init__(self, x, y, role_image_path, image_size, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.role_name = role_image_path[0]
		self.base_image = pygame.image.load(role_image_path[1]).convert()
		self.base_image = pygame.transform.scale(self.base_image, image_size)
		self.image = self.base_image.copy()
		self.rect = self.image.get_rect()
		self.rect.center = (x, y)
		self.base_speed = [16, 16]
		self.speed = [0, 0]
	'''update'''
	def update(self, direction, wall_sprites, gate_sprites):
		# update attributes
		if direction[0] < 0:
			self.image = pygame.transform.flip(self.base_image, True, False)
		elif direction[0] > 0:
			self.image = self.base_image.copy()
		elif direction[1] < 0:
			self.image = pygame.transform.rotate(self.base_image, 90)
		elif direction[1] > 0:
			self.image = pygame.transform.rotate(self.base_image, -90)
		self.speed = [direction[0] * self.base_speed[0], direction[1] * self.base_speed[1]]
		# try move
		x_prev = self.rect.left
		y_prev = self.rect.top
		self.rect.left += self.speed[0]
		self.rect.top += self.speed[1]
		is_collide = pygame.sprite.spritecollide(self, wall_sprites, False)
		if gate_sprites is not None:
			if not is_collide:
				is_collide = pygame.sprite.spritecollide(self, gate_sprites, False)
		if is_collide:
			self.rect.left = x_prev
			self.rect.top = y_prev
			return False
		return True