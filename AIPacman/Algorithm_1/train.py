'''
Function:
	train the model
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import config
from nets.nets import DQNet, DQNAgent
from gameAPI.game import GamePacmanAgent


'''train the model'''
def train():
	game_pacman_agent = GamePacmanAgent(config)
	dqn_net = DQNet(True, config)
	dqn_agent = DQNAgent(game_pacman_agent, dqn_net, config)
	dqn_agent.train()


'''run'''
if __name__ == '__main__':
	train()