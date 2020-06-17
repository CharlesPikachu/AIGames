'''
Function:
	use ai to play greedy snake
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cfg
import pygame
from modules.food import *
from modules.snake import *
from modules.utils import *
from modules.agent import *
from modules.endInterface import *


'''main function'''
def main(cfg):
	# initialize the game
	pygame.init()
	screen = pygame.display.set_mode(cfg.SCREENSIZE)
	pygame.display.set_caption('AI Snake —— 微信公众号:Charles的皮卡丘')
	clock = pygame.time.Clock()
	# play the background music
	pygame.mixer.music.load(cfg.BGMPATH)
	pygame.mixer.music.play(-1)
	# the game main loop
	snake = Snake(cfg)
	ai = Agent(cfg, snake)
	apple = Apple(cfg, snake.coords)
	score = 0
	while True:
		screen.fill(cfg.BLACK)
		# --check the keyboard
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
		# --make decision by agent
		infos = ai.act(snake, apple)
		# --update the food and score
		if infos and infos['eaten']:
			apple = infos['food']
			assert apple, 'bugs may exist'
			score += 1
		# --judge whether the game is over
		if snake.isgameover: break
		# --draw the necessary elements in the game
		drawGameGrid(cfg, screen)
		snake.draw(screen)
		apple.draw(screen)
		showScore(cfg, score, screen)
		# --update the screen
		pygame.display.update()
		clock.tick(cfg.FPS)
	return endInterface(screen, cfg)


'''run'''
if __name__ == '__main__':
	while True:
		try:
			if not main(cfg):
				break
		except:
			continue