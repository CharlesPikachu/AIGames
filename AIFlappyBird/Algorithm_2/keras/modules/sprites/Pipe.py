'''
Function:
	Define the pipe class.
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import random
import pygame


'''pipe class'''
class Pipe(pygame.sprite.Sprite):
	def __init__(self, image, position, type_, **kwargs):
		pygame.sprite.Sprite.__init__(self)
		self.image = image
		self.rect = self.image.get_rect()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect.left, self.rect.top = position
		self.type_ = type_
		self.used_for_score = False
	@staticmethod
	def randomPipe(cfg, image):
		base_y = 0.79 * cfg.SCREENHEIGHT
		up_y = int(base_y * 0.2) + random.randrange(0, int(base_y * 0.6 - cfg.PIPE_GAP_SIZE))
		return {'top': (cfg.SCREENWIDTH+10, up_y-image.get_height()), 'bottom': (cfg.SCREENWIDTH+10, up_y+cfg.PIPE_GAP_SIZE)}