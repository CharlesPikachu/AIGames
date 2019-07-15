'''
Function:
	define the game
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cv2
import random
import pygame


'''game of pong'''
class PongGame():
	def __init__(self):
		self.__initGame()
	'''update frame and return the necessary data, action: [up, still, down]'''
	def nextFrame(self, action):
		pygame.event.pump()
		reward = 0
		# paddle 1 action
		if action[0] == 1:
			paddle_1_speed = -self.paddle_speed
		elif action[1] == 1:
			paddle_1_speed = 0
		elif action[2] == 1:
			paddle_1_speed = self.paddle_speed
		self.paddle_1_pos = self.paddle_1_pos[0], max(min(paddle_1_speed + self.paddle_1_pos[1], 420), 10)
		# paddle 2 action (just an easy ai)
		if self.ball_pos[0] >= 305.:
			if not self.paddle_2_pos[1] == self.ball_pos[1] + 7.5:
				if self.paddle_2_pos[1] < self.ball_pos[1] + 7.5:
					paddle_2_speed = self.paddle_speed
					self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + paddle_2_speed, 420), 10)
				if self.paddle_2_pos[1] > self.ball_pos[1] - 42.5:
					paddle_2_speed = -self.paddle_speed
					self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + paddle_2_speed, 420), 10)
			else:
				self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + 7.5, 420), 10)
		# ball action
		if self.ball_pos[0] <= self.paddle_1_pos[0] + 10.:
			if self.ball_pos[1] + 7.5 >= self.paddle_1_pos[1] and self.ball_pos[1] <= self.paddle_1_pos[1] + 42.5:
				self.ball_pos = 20., self.ball_pos[1]
				self.ball_speed = self.ball_speed_base[0], self.ball_speed_base[1] * random.choice([-1, 1])
				reward = 1
		if self.ball_pos[0] + 15 >= self.paddle_2_pos[0]:
			if self.ball_pos[1] + 7.5 >= self.paddle_2_pos[1] and self.ball_pos[1] <= self.paddle_2_pos[1] + 42.5:
				self.ball_pos = 605., self.ball_pos[1]
				self.ball_speed = -self.ball_speed_base[0], self.ball_speed_base[1] * random.choice([-1, 1])
		if self.ball_pos[0] < 5.:
			self.paddle_2_score += 1
			reward = -2
			self.paddle_1_pos = (10., 215.)
			self.paddle_2_pos = (620., 215.)
			self.ball_pos = (312.5, 232.5)
		elif self.ball_pos[0] > 620.:
			self.paddle_1_score += 1
			reward = 2
			self.paddle_1_pos = (10., 215.)
			self.paddle_2_pos = (620., 215.)
			self.ball_pos = (312.5, 232.5)
		if self.ball_pos[1] <= 10.:
			self.ball_speed = self.ball_speed[0], -self.ball_speed[1]
			self.ball_pos = self.ball_pos[0], 10
		elif self.ball_pos[1] >= 455:
			self.ball_speed = self.ball_speed[0], -self.ball_speed[1]
			self.ball_pos = self.ball_pos[0], 455
		self.ball_pos = self.ball_pos[0] + self.ball_speed[0], self.ball_pos[1] + self.ball_speed[1]
		# show the sprites
		self.screen.blit(self.background, (0, 0))
		pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect((5, 5), (630, 470)), 2)
		pygame.draw.aaline(self.screen, (255, 255, 255), (320, 5), (320, 475))
		self.screen.blit(self.paddle_1, self.paddle_1_pos)
		self.screen.blit(self.paddle_2, self.paddle_2_pos)
		self.screen.blit(self.ball, self.ball_pos)
		# get screenshot
		frame = pygame.surfarray.array3d(pygame.display.get_surface())
		frame = cv2.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		# show the score
		score1_render = self.font.render(str(self.paddle_1_score), True, (255, 255, 255))
		score2_render = self.font.render(str(self.paddle_2_score), True, (255, 255, 255))
		self.screen.blit(score1_render, (240, 210))
		self.screen.blit(score2_render, (370, 210))
		# update
		pygame.display.update()
		paddle_1_score = self.paddle_1_score
		paddle_2_score = self.paddle_2_score
		if self.paddle_1_score >= self.win_score:
			self.__reset()
			reward = 10
			terminal = True
		elif self.paddle_2_score >= self.win_score:
			self.__reset()
			reward = -10
			terminal = True
		else:
			terminal = False
		# return the necessary data
		return frame, action, reward, terminal, paddle_1_score, paddle_2_score
	'''game reset'''
	def __reset(self):
		self.__initGame()
	'''game initialize'''
	def __initGame(self):
		pygame.init()
		self.screen = pygame.display.set_mode((640, 480), 0, 32)
		self.background = pygame.Surface((640, 480)).convert()
		self.background.fill((0, 0, 0))
		# paddle
		self.paddle_speed = 15
		self.paddle_1_score = 0
		# --paddle1 → for train
		self.paddle_1 = pygame.Surface((10, 50)).convert()
		self.paddle_1.fill((0, 255, 255))
		self.paddle_1_pos = (10., 215.)
		# --paddle2 → an easy ai
		self.paddle_2 = pygame.Surface((10, 50)).convert()
		self.paddle_2.fill((255, 255, 0))
		self.paddle_2_pos = (620., 215.)
		# ball
		self.ball_speed_base = (7, 7)
		self.ball_speed = (7, 7)
		self.paddle_2_score = 0
		self.ball = pygame.Surface((16, 16)).convert()
		self.ball.set_colorkey((0, 0, 0))
		pygame.draw.circle(self.ball, (255, 255, 255), (7, 7), (7))
		self.ball_pos = (312.5, 232.5)
		# font
		self.font = pygame.font.SysFont("calibri", 40)
		# score to win
		self.win_score = 20