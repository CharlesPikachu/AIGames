'''config file'''
import os


'''the resource paths'''
BGMPATH = os.path.join(os.getcwd(), 'resources/music/bgm.mp3')
FONTPATH = os.path.join(os.getcwd(), 'resources/font/Gabriola.ttf')
'''screen size'''
SCREENSIZE = (400, 400)
'''FPS'''
FPS = 30
'''some constants'''
BLOCK_SIZE = 20
BLACK = (0, 0, 0)
GAME_MATRIX_SIZE = (int(SCREENSIZE[0]/BLOCK_SIZE), int(SCREENSIZE[1]/BLOCK_SIZE))