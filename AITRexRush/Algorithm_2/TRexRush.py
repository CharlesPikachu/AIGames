'''
Function:
	use image recognition algorithm to play TRexRush
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import cfg
from modules.agent.agent import Agent
from modules.gameapis.controller import GameController


'''main'''
def main():
	# the instanced class of GameController
	game_cotroller = GameController(cfg)
	# the instanced class of Agent
	bbox_area = (80, 100, 100, 120)
	agent = Agent(bbox_area)
	# play TRexRush with our agent
	image, score, is_dead = game_cotroller.run([1, 0], bbox_area)
	while True:
		action = agent.act(image)
		image, score, is_dead = game_cotroller.run(action, bbox_area)


'''run'''
if __name__ == '__main__':
	main()