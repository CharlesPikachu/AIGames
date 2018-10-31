'''
乒乓球小游戏
作者: Charles
公众号: Charles的皮卡丘
'''
import pygame


'''
乒乓球游戏类
'''
class PongGame():
	def __init__(self):
		self.__initGame()
		# 初始化一些变量
		self.loseReward = -1
		self.winReward = 1
		self.hitReward = 0
		self.paddleSpeed = 15
		self.ballSpeed = (7, 7)
		self.paddle_1_score = 0
		self.paddle_2_score = 0
		self.paddle_1_speed = 0.
		self.paddle_2_speed = 0.
		self.__reset()
	'''
	更新一帧
	action: [keep, up, down]
	'''
	def update_frame(self, action):
		assert len(action) == 3
		pygame.event.pump()
		reward = 0
		# 绑定一些对象
		self.score1Render = self.font.render(str(self.paddle_1_score), True, (255, 255, 255))
		self.score2Render = self.font.render(str(self.paddle_2_score), True, (255, 255, 255))
		self.screen.blit(self.background, (0, 0))
		pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect((5, 5), (630, 470)), 2)
		pygame.draw.aaline(self.screen, (255, 255, 255), (320, 5), (320, 475))
		self.screen.blit(self.paddle_1, self.paddle_1_pos)
		self.screen.blit(self.paddle_2, self.paddle_2_pos)
		self.screen.blit(self.ball, self.ball_pos)
		self.screen.blit(self.score1Render, (240, 210))
		self.screen.blit(self.score2Render, (370, 210))
		# 行动paddle_1(训练对象)
		if action[0] == 1:
			self.paddle_1_speed = 0
		elif action[1] == 1:
			self.paddle_1_speed = -self.paddleSpeed
		elif action[2] == 1:
			self.paddle_1_speed = self.paddleSpeed
		self.paddle_1_pos = self.paddle_1_pos[0], max(min(self.paddle_1_speed + self.paddle_1_pos[1], 420), 10)
		# 行动paddle_2(设置一个简单的算法使paddle_2的表现较优, 非训练对象)
		if self.ball_pos[0] >= 305.:
			if not self.paddle_2_pos[1] == self.ball_pos[1] + 7.5:
				if self.paddle_2_pos[1] < self.ball_pos[1] + 7.5:
					self.paddle_2_speed = self.paddleSpeed
					self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + self.paddle_2_speed, 420), 10)
				if self.paddle_2_pos[1] > self.ball_pos[1] - 42.5:
					self.paddle_2_speed = -self.paddleSpeed
					self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + self.paddle_2_speed, 420), 10)
			else:
				self.paddle_2_pos = self.paddle_2_pos[0], max(min(self.paddle_2_pos[1] + 7.5, 420), 10)
		# 行动ball
		# 	球撞拍上
		if self.ball_pos[0] <= self.paddle_1_pos[0] + 10.:
			if self.ball_pos[1] + 7.5 >= self.paddle_1_pos[1] and self.ball_pos[1] <= self.paddle_1_pos[1] + 42.5:
				self.ball_pos = 20., self.ball_pos[1]
				self.ballSpeed = -self.ballSpeed[0], self.ballSpeed[1]
				reward = self.hitReward
		if self.ball_pos[0] + 15 >= self.paddle_2_pos[0]:
			if self.ball_pos[1] + 7.5 >= self.paddle_2_pos[1] and self.ball_pos[1] <= self.paddle_2_pos[1] + 42.5:
				self.ball_pos = 605., self.ball_pos[1]
				self.ballSpeed = -self.ballSpeed[0], self.ballSpeed[1]
		# 	拍未接到球(另外一个拍得分)
		if self.ball_pos[0] < 5.:
			self.paddle_2_score += 1
			reward = self.loseReward
			self.__reset()
		elif self.ball_pos[0] > 620.:
			self.paddle_1_score += 1
			reward = self.winReward
			self.__reset()
		# 	球撞墙上
		if self.ball_pos[1] <= 10.:
			self.ballSpeed = self.ballSpeed[0], -self.ballSpeed[1]
			self.ball_pos = self.ball_pos[0], 10
		elif self.ball_pos[1] >= 455:
			self.ballSpeed = self.ballSpeed[0], -self.ballSpeed[1]
			self.ball_pos = self.ball_pos[0], 455
		# 	更新ball的位置
		self.ball_pos = self.ball_pos[0] + self.ballSpeed[0], self.ball_pos[1] + self.ballSpeed[1]
		# 获取当前场景(只取左半边)
		image = pygame.surfarray.array3d(pygame.display.get_surface())
		# image = image[321:, :]
		pygame.display.update()
		terminal = False
		if max(self.paddle_1_score, self.paddle_2_score) >= 20:
			self.paddle_1_score = 0
			self.paddle_2_score = 0
			terminal = True
		return image, reward, terminal
	'''
	游戏初始化
	'''
	def __initGame(self):
		pygame.init()
		self.screen = pygame.display.set_mode((640, 480), 0, 32)
		self.background = pygame.Surface((640, 480)).convert()
		self.background.fill((0, 0, 0))
		self.paddle_1 = pygame.Surface((10, 50)).convert()
		self.paddle_1.fill((0, 255, 255))
		self.paddle_2 = pygame.Surface((10, 50)).convert()
		self.paddle_2.fill((255, 255, 0))
		ball_surface = pygame.Surface((15, 15))
		pygame.draw.circle(ball_surface, (255, 255, 255), (7, 7), (7))
		self.ball = ball_surface.convert()
		self.ball.set_colorkey((0, 0, 0))
		self.font = pygame.font.SysFont("calibri", 40)
	'''
	重置球和球拍的位置
	'''
	def __reset(self):
		self.paddle_1_pos = (10., 215.)
		self.paddle_2_pos = (620., 215.)
		self.ball_pos = (312.5, 232.5)