'''
Function:
	use dqn to play TRexRush
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import os
import cfg
import argparse
from modules.DQNAgent.agent import DQNAgent
from modules.gameapis.controller import GameController


'''parse arguments'''
def parseArgs():
	parser = argparse.ArgumentParser(description='Use dpn to play TRexRush')
	parser.add_argument('--mode', dest='mode', help='Choose <train> or <test> please', default='train', type=str)
	parser.add_argument('--resume', dest='resume', help='If mode is <train> and use --resume, check and load the training history', action='store_true')
	args = parser.parse_args()
	return args


'''main'''
def main():
	# parse arguments in command line
	args = parseArgs()
	mode = args.mode.lower()
	assert mode in ['train', 'test'], '--mode should be <train> or <test>'
	# the instanced class of DQNAgent, and the path to save and load model
	if not os.path.exists('checkpoints'):
		os.mkdir('checkpoints')
	checkpointspath = 'checkpoints/dqn.pth'
	agent = DQNAgent(mode=mode, fps=cfg.FPS, checkpointspath=checkpointspath)
	if os.path.isfile(checkpointspath):
		if mode == 'test' or (args.resume and mode == 'train'):
			agent.load(checkpointspath)
	# the instanced class of GameController
	game_cotroller = GameController(cfg)
	# begin game
	if mode == 'train':
		agent.train(game_cotroller)
	else:
		agent.test(game_cotroller)


'''run'''
if __name__ == '__main__':
	main()