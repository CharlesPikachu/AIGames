'''
Function:
	use genetic algorithm to play google's t-rex rush
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cfg
import sys
import random
import pygame
import argparse
from modules.agent.agent import *
from modules.sprites.scene import *
from modules.sprites.obstacle import *
from modules.interfaces.gameend import GameEndInterface
from modules.interfaces.gamestart import GameStartInterface


'''parse arguments'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Use genetic algorithm to play TRexRush')
	parser.add_argument('--resume', dest='resume', help='whether load the checkpoints or not', action='store_true')
	args = parser.parse_args()
	return args


'''main'''
def main(highest_score, args):
	# initialize the game
	pygame.init()
	screen = pygame.display.set_mode(cfg.SCREENSIZE)
	pygame.display.set_caption('T-Rex Rush —— Charles的皮卡丘')
	# load all audios
	sounds = {}
	for key, value in cfg.AUDIO_PATHS.items():
		sounds[key] = pygame.mixer.Sound(value)
	# the game start interface
	GameStartInterface(screen, sounds, cfg)
	# the ai agent to control dinos
	dinos_agent = Agent(cfg, sounds)
	if args.resume: dinos_agent.load()
	save_interval = 1
	# always run the game for training
	while True:
		# define the necessary elements in our game and some essential variables
		score = 0
		score_board = Scoreboard(cfg.IMAGE_PATHS['numbers'], position=(534, 15), bg_color=cfg.BACKGROUND_COLOR)
		highest_score = highest_score
		highest_score_board = Scoreboard(cfg.IMAGE_PATHS['numbers'], position=(435, 15), bg_color=cfg.BACKGROUND_COLOR, is_highest=True)
		ground = Ground(cfg.IMAGE_PATHS['ground'], position=(0, cfg.SCREENSIZE[1]))
		cloud_sprites_group = pygame.sprite.Group()
		cactus_sprites_group = pygame.sprite.Group()
		ptera_sprites_group = pygame.sprite.Group()
		add_obstacle_timer = 0
		score_timer = 0
		# the main loop of our game
		clock = pygame.time.Clock()
		dinos_agent.num_iter += 1
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
			screen.fill(cfg.BACKGROUND_COLOR)
			# --make decision for each dino, 84 means dino.rect.right when dino.is_ducking = False
			if len(cactus_sprites_group) > 0 or len(ptera_sprites_group) > 0:
				nearest_obstacle = None
				for item in cactus_sprites_group:
					if item.rect.left < 84:
						continue
					if nearest_obstacle is None:
						nearest_obstacle = item
					else:
						if item.rect.left < nearest_obstacle.rect.left: nearest_obstacle = item
				for item in ptera_sprites_group:
					if item.rect.left < 84:
						continue
					if nearest_obstacle is None:
						nearest_obstacle = item
					else:
						if item.rect.left < nearest_obstacle.rect.left: nearest_obstacle = item
				if nearest_obstacle:
					inputs = [nearest_obstacle.rect.left-84, nearest_obstacle.rect.bottom, nearest_obstacle.rect.width, nearest_obstacle.rect.height, -ground.speed]
					dinos_agent.makedecision(inputs)
			# --add cloud randomly
			if len(cloud_sprites_group) < 5 and random.randrange(0, 300) == 10:
				cloud_sprites_group.add(Cloud(cfg.IMAGE_PATHS['cloud'], position=(cfg.SCREENSIZE[0], random.randrange(30, 75))))
			# --add obstacles randomly
			add_obstacle_timer += 1
			if add_obstacle_timer > random.randrange(50, 150):
				add_obstacle_timer = 0
				random_value = random.randrange(0, 10)
				if random_value >= 5 and random_value <= 7:
					cactus_sprites_group.add(Cactus(cfg.IMAGE_PATHS['cacti']))
				else:
					position_ys = [cfg.SCREENSIZE[1]*0.82, cfg.SCREENSIZE[1]*0.75, cfg.SCREENSIZE[1]*0.60, cfg.SCREENSIZE[1]*0.20]
					ptera_sprites_group.add(Ptera(cfg.IMAGE_PATHS['ptera'], position=(600, random.choice(position_ys))))
			# --update the game elements
			dinos_agent.update()
			ground.update()
			cloud_sprites_group.update()
			cactus_sprites_group.update()
			ptera_sprites_group.update()
			score_timer += 1
			if score_timer > (cfg.FPS//12):
				score_timer = 0
				score += 1
				score = min(score, 99999)
				if score > highest_score:
					highest_score = score
				if score % 100 == 0:
					sounds['point'].play()
				if score % 1000 == 0:
					ground.speed -= 1
					for item in cloud_sprites_group:
						item.speed -= 1
					for item in cactus_sprites_group:
						item.speed -= 1
					for item in ptera_sprites_group:
						item.speed -= 1
			# --collision detection
			for cacti in cactus_sprites_group:
				for dino in dinos_agent.dinos:
					if dino.is_dead: continue
					if pygame.sprite.collide_mask(dino, cacti):
						dino.die(sounds)
			for ptera in ptera_sprites_group:
				for dino in dinos_agent.dinos:
					if dino.is_dead: continue
					if pygame.sprite.collide_mask(dino, ptera):
						dino.die(sounds)
			# --draw the game elements on the screen
			dinos_agent.draw(screen)
			ground.draw(screen)
			cloud_sprites_group.draw(screen)
			cactus_sprites_group.draw(screen)
			ptera_sprites_group.draw(screen)
			score_board.set(score)
			highest_score_board.set(highest_score)
			score_board.draw(screen)
			highest_score_board.draw(screen)
			# --update the screen
			pygame.display.update()
			clock.tick(cfg.FPS)
			# --update ai model
			num_dies = 0
			num_alives = 0
			for dino in dinos_agent.dinos:
				if dino.is_dead:
					num_dies += 1
					continue
				num_alives += 1
				dino.score = score
			print(f'[Iter]: {dinos_agent.num_iter}, [Score]: {score}, [Max Score]: {highest_score}, [Dead dino]: {num_dies}, [Alive dino]: {num_alives}')
			if num_alives == 0:
				break
		if dinos_agent.num_iter % save_interval == 0:
			dinos_agent.save()
		dinos_agent.nextgeneration()
	# the game end interface
	return GameEndInterface(screen, cfg), highest_score


'''run'''
if __name__ == '__main__':
	args = parseArgs()
	highest_score = 0
	while True:
		flag, highest_score = main(highest_score, args)
		if not flag: break