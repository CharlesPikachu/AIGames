'''
Function:
	Define the bird class.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import pygame
import itertools


'''bird class'''
class Bird(pygame.sprite.Sprite):
	def __init__(self, images, idx, position, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.images = images
		self.image = list(images.values())[idx]
		self.rect = self.image.get_rect()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect.left, self.rect.top = position
		# variables required for vertical movement
		self.is_flapped = False
		self.speed = -9
		# variables required for bird status switch
		self.bird_idx = idx
		self.bird_idx_cycle = itertools.cycle([0, 1, 2, 1])
		self.bird_idx_change_count = 0
	'''update bird'''
	def update(self, boundary_values):
		# update the position vertically
		if not self.is_flapped:
			self.speed = min(self.speed+1, 10)
		self.is_flapped = False
		self.rect.top += self.speed
		# determine if the bird dies because it hits the upper and lower boundaries
		is_dead = False
		if self.rect.bottom > boundary_values[1]:
			is_dead = True
			self.rect.bottom = boundary_values[1]
		if self.rect.top < boundary_values[0]:
			self.rect.top = boundary_values[0]
		# simulate wing vibration
		self.bird_idx_change_count += 1
		if self.bird_idx_change_count % 3 == 0:
			self.bird_idx = next(self.bird_idx_cycle)
			self.image = list(self.images.values())[self.bird_idx]
			self.bird_idx_change_count = 0
		return is_dead
	'''set to fly mode'''
	def setFlapped(self):
		self.is_flapped = True
		self.speed = -9
	'''bind to screen'''
	def draw(self, screen):
		screen.blit(self.image, self.rect)