'''
Function:
	AI贪吃蛇
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import sys
import random
import pygame


'''错误码'''
ERR = -404
'''屏幕大小'''
SCREENWIDTH = 800
SCREENHEIGHT = 500
'''刷新频率'''
FPS = 17
'''一块蛇身大小'''
CELLSIZE = 20
assert SCREENWIDTH % CELLSIZE == 0
assert SCREENHEIGHT % CELLSIZE == 0
'''等价的运动区域大小'''
MATRIX_W = int(SCREENWIDTH/CELLSIZE)
MATRIX_H = int(SCREENHEIGHT/CELLSIZE)
MATRIX_SIZE = MATRIX_W * MATRIX_H
'''背景颜色'''
BGCOLOR = (0, 0, 0)
'''蛇头索引'''
HEADINDEX = 0
'''最佳运动方向'''
BESTMOVE = ERR
'''不同东西在矩阵里用不同的数字表示'''
FOODNUM = 0
SPACENUM = (MATRIX_W + 1) * (MATRIX_H + 1)
SNAKENUM = 2 * SPACENUM
'''运动方向字典'''
MOVEDIRECTIONS = {
					'left': -1,
					'right': 1,
					'up': -MATRIX_W,
					'down': MATRIX_W
					}


'''关闭游戏界面'''
def CloseGame():
	pygame.quit()
	sys.exit()


'''显示当前得分'''
def ShowScore(score):
	score_render = default_font.render('得分: %s' % (score), True, (255, 255, 255))
	rect = score_render.get_rect()
	rect.topleft = (SCREENWIDTH-120, 10)
	screen.blit(score_render, rect)


'''获得果实位置'''
def GetAppleLocation(snake_coords):
	flag = True
	while flag:
		apple_location = {'x': random.randint(0, MATRIX_W-1), 'y': random.randint(0, MATRIX_H-1)}
		if apple_location not in snake_coords:
			flag = False
	return apple_location


'''显示果实'''
def ShowApple(coord):
	x = coord['x'] * CELLSIZE
	y = coord['y'] * CELLSIZE
	rect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
	pygame.draw.rect(screen, (255, 0, 0), rect)


'''显示蛇'''
def ShowSnake(coords):
	x = coords[0]['x'] * CELLSIZE
	y = coords[0]['y'] * CELLSIZE
	head_rect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
	pygame.draw.rect(screen, (0, 80, 255), head_rect)
	head_inner_rect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
	pygame.draw.rect(screen, (0, 80, 255), head_inner_rect)
	for coord in coords[1:]:
		x = coord['x'] * CELLSIZE
		y = coord['y'] * CELLSIZE
		rect = pygame.Rect(x, y, CELLSIZE, CELLSIZE)
		pygame.draw.rect(screen, (0, 155, 0), rect)
		inner_rect = pygame.Rect(x + 4, y + 4, CELLSIZE - 8, CELLSIZE - 8)
		pygame.draw.rect(screen, (0, 255, 0), inner_rect)


'''画网格'''
def drawGrid():
	# 垂直方向
	for x in range(0, SCREENWIDTH, CELLSIZE):
		pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, SCREENHEIGHT))
	# 水平方向
	for y in range(0, SCREENHEIGHT, CELLSIZE):
		pygame.draw.line(screen, (40, 40, 40), (0, y), (SCREENWIDTH, y))


'''显示结束界面'''
def ShowEndInterface():
	title_font = pygame.font.Font('simkai.ttf', 100)
	title_game = title_font.render('Game', True, (233, 150, 122))
	title_over = title_font.render('Over', True, (233, 150, 122))
	game_rect = title_game.get_rect()
	over_rect = title_over.get_rect()
	game_rect.midtop = (SCREENWIDTH/2, 70)
	over_rect.midtop = (SCREENWIDTH/2, game_rect.height+70+25)
	screen.blit(title_game, game_rect)
	screen.blit(title_over, over_rect)
	pygame.display.update()
	pygame.time.wait(500)
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
				CloseGame()


'''判断该位置是否为空'''
def IsCellFree(idx, psnake):
	location_x = idx % MATRIX_W
	location_y = idx // MATRIX_W
	idx = {'x': location_x, 'y': location_y}
	return (idx not in psnake)


'''重置board'''
def ResetBoard(psnake, pboard, pfood):
	temp_board = pboard[:]
	pfood_idx = pfood['x'] + pfood['y'] * MATRIX_W
	for i in range(MATRIX_SIZE):
		if i == pfood_idx:
			temp_board[i] = FOODNUM
		elif IsCellFree(i, psnake):
			temp_board[i] = SPACENUM
		else:
			temp_board[i] = SNAKENUM
	return temp_board


'''检查位置idx是否可以向当前move方向运动'''
def isMovePossible(idx, move_direction):
	flag = False
	if move_direction == 'left':
		if idx % MATRIX_W > 0: flag = True
		else: flag = False
	elif move_direction == 'right':
		if idx % MATRIX_W < MATRIX_W - 1: flag = True
		else: flag = False
	elif move_direction == 'up':
		if idx > MATRIX_W - 1: flag = True
		else: flag = False
	elif move_direction == 'down':
		if idx < MATRIX_SIZE - MATRIX_W: flag = True
		else: flag = False
	return flag


'''广度优先搜索遍历整个board, 计算出board中每个非SNAKENUM元素到达食物的路径长度'''
def RefreshBoard(psnake, pfood, pboard):
	temp_board = pboard[:]
	pfood_idx = pfood['x'] + pfood['y'] * MATRIX_W
	queue = []
	queue.append(pfood_idx)
	inqueue = [0] * MATRIX_SIZE
	found = False
	while len(queue) != 0:
		idx = queue.pop(0)
		if inqueue[idx] == 1:
			continue
		inqueue[idx] = 1
		for move_direction in ['left', 'right', 'up', 'down']:
			if isMovePossible(idx, move_direction):
				if (idx + MOVEDIRECTIONS[move_direction]) == (psnake[HEADINDEX]['x'] + psnake[HEADINDEX]['y'] * MATRIX_W):
					found = True
				# 该点不是蛇身(食物是0才可以这样子写)
				if temp_board[idx + MOVEDIRECTIONS[move_direction]] < SNAKENUM:
					if temp_board[idx + MOVEDIRECTIONS[move_direction]] > temp_board[idx]+1:
						temp_board[idx + MOVEDIRECTIONS[move_direction]] = temp_board[idx] + 1
					if inqueue[idx + MOVEDIRECTIONS[move_direction]] == 0:
						queue.append(idx + MOVEDIRECTIONS[move_direction])
	return (found, temp_board)


'''根据board中元素值, 从蛇头周围4个领域点中选择最短路径'''
def chooseShortestSafeMove(psnake, pboard):
	BESTMOVE = ERR
	min_distance = SNAKENUM
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[HEADINDEX]['x'] + psnake[HEADINDEX]['y'] * MATRIX_W
		if isMovePossible(idx, move_direction) and (pboard[idx + MOVEDIRECTIONS[move_direction]] < min_distance):
			min_distance = pboard[idx + MOVEDIRECTIONS[move_direction]]
			BESTMOVE = move_direction
	return BESTMOVE


'''找到移动后蛇头的位置'''
def findSnakeHead(snake_coords, direction):
	if direction == 'up':
		new_head = {'x': snake_coords[HEADINDEX]['x'],
					'y': snake_coords[HEADINDEX]['y'] - 1}
	elif direction == 'down':
		new_head = {'x': snake_coords[HEADINDEX]['x'],
					'y': snake_coords[HEADINDEX]['y'] + 1}
	elif direction == 'left':
		new_head = {'x': snake_coords[HEADINDEX]['x'] - 1,
					'y': snake_coords[HEADINDEX]['y']}
	elif direction == 'right':
		new_head = {'x': snake_coords[HEADINDEX]['x'] + 1,
					'y': snake_coords[HEADINDEX]['y']}
	return new_head


'''虚拟地运行一次'''
def virtualMove(psnake, pboard, pfood):
	temp_snake = psnake[:]
	temp_board = pboard[:]
	reset_tboard = ResetBoard(temp_snake, temp_board, pfood)
	temp_board = reset_tboard
	food_eated = False
	while not food_eated:
		refresh_tboard = RefreshBoard(temp_snake, pfood, temp_board)[1]
		temp_board = refresh_tboard
		move_direction = chooseShortestSafeMove(temp_snake, temp_board)
		snake_coords = temp_snake[:]
		temp_snake.insert(0, findSnakeHead(snake_coords, move_direction))
		# 如果新的蛇头正好是食物的位置
		if temp_snake[HEADINDEX] == pfood:
			reset_tboard = ResetBoard(temp_snake, temp_board, pfood)
			temp_board = reset_tboard
			pfood_idx = pfood['x'] + pfood['y'] * MATRIX_W
			temp_board[pfood_idx] = SNAKENUM
			food_eated = True
		else:
			new_head_idx = temp_snake[0]['x'] + temp_snake[0]['y'] * MATRIX_W
			temp_board[new_head_idx] = SNAKENUM
			end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * MATRIX_W
			temp_board[end_idx] = SPACENUM
			del temp_snake[-1]
	return temp_snake, temp_board


'''检查蛇头和蛇尾间是有路径的, 避免蛇陷入死路'''
def isTailInside(psnake, pboard, pfood):
	temp_board = pboard[:]
	temp_snake = psnake[:]
	# 将蛇尾看作食物
	end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * MATRIX_W
	temp_board[end_idx] = FOODNUM
	v_food = temp_snake[-1]
	# 食物看作蛇身(重复赋值了)
	pfood_idx = pfood['x'] + pfood['y'] * MATRIX_W
	temp_board[pfood_idx] = SNAKENUM
	# 求得每个位置到蛇尾的路径长度
	result, refresh_tboard = RefreshBoard(temp_snake, v_food, temp_board)
	temp_board = refresh_tboard
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = temp_snake[HEADINDEX]['x'] + temp_snake[HEADINDEX]['y'] * MATRIX_W
		end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * MATRIX_W
		if isMovePossible(idx, move_direction) and (idx + MOVEDIRECTIONS[move_direction] == end_idx) and (len(temp_snake) > 3):
			result = False
	return result


'''根据board中元素值, 从蛇头周围4个领域点中选择最远路径'''
def chooseLongestSafeMove(psnake, pboard):
	BESTMOVE = ERR
	max_distance = -1
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[HEADINDEX]['x'] + psnake[HEADINDEX]['y'] * MATRIX_W
		if isMovePossible(idx, move_direction) and (pboard[idx + MOVEDIRECTIONS[move_direction]] > max_distance) and (pboard[idx + MOVEDIRECTIONS[move_direction]] < SPACENUM):
			max_distance = pboard[idx + MOVEDIRECTIONS[move_direction]]
			BESTMOVE = move_direction
	return BESTMOVE 


'''让蛇头朝着蛇尾运行一步'''
def followTail(psnake, pboard, pfood):
	temp_snake = psnake[:]
	temp_board = ResetBoard(temp_snake, pboard, pfood)
	# 将蛇尾看作食物
	end_idx = temp_snake[-1]['x'] + temp_snake[-1]['y'] * MATRIX_W
	temp_board[end_idx] = FOODNUM
	v_food = temp_snake[-1]
	# 食物看作蛇身
	pfood_idx = pfood['x'] + pfood['y'] * MATRIX_W
	temp_board[pfood_idx] = SNAKENUM
	# 求得每个位置到蛇尾的路径长度
	result, refresh_tboard = RefreshBoard(temp_snake, v_food, temp_board)
	temp_board = refresh_tboard
	# 还原
	temp_board[end_idx] = SNAKENUM
	# temp_board[pfood_idx] = FOOD
	return chooseLongestSafeMove(temp_snake, temp_board)


'''如果蛇和食物间有路径, 则需要找一条安全的路径'''
def findSafeWay(psnake, pboard, pfood):
	safe_move = ERR
	real_snake = psnake[:]
	real_board = pboard[:]
	v_psnake, v_pboard = virtualMove(psnake, pboard, pfood)
	# 如果虚拟运行后，蛇头蛇尾间有通路，则选最短路运行
	if isTailInside(v_psnake, v_pboard, pfood):
		safe_move = chooseShortestSafeMove(real_snake, real_board)
	else:
		safe_move = followTail(real_snake, real_board, pfood)
	return safe_move


'''各种方案均无效时，随便走一步'''
def anyPossibleMove(psnake, pboard, pfood):
	BESTMOVE = ERR
	reset_board = ResetBoard(psnake, pboard, pfood)
	pboard = reset_board
	result, refresh_board = RefreshBoard(psnake, pfood, pboard)
	pboard = refresh_board
	min_distance = SNAKENUM
	for move_direction in ['left', 'right', 'up', 'down']:
		idx = psnake[HEADINDEX]['x'] + psnake[HEADINDEX]['y'] * MATRIX_W
		if isMovePossible(idx, move_direction) and (pboard[idx + MOVEDIRECTIONS[move_direction]]<min_distance):
			min_distance = pboard[idx + MOVEDIRECTIONS[move_direction]]
			BESTMOVE = move_direction
	return BESTMOVE


'''运行游戏'''
def RunGame():
	# 一维数组来表示蛇运动的矩形场地
	board = [0] * MATRIX_SIZE
	# 蛇出生地
	start_x = random.randint(5, MATRIX_W-6)
	start_y = random.randint(5, MATRIX_H-6)
	snake_coords = [{'x': start_x, 'y': start_y},
					{'x': start_x-1, 'y': start_y},
					{'x': start_x-2, 'y': start_y}]
	apple_location = GetAppleLocation(snake_coords)
	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				CloseGame()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					CloseGame()
		screen.fill(BGCOLOR)
		drawGrid()
		ShowSnake(snake_coords)
		ShowApple(apple_location)
		ShowScore(len(snake_coords) - 3)
		# 重置board
		reset_board = ResetBoard(snake_coords, board, apple_location)
		board = reset_board
		result, refresh_board = RefreshBoard(snake_coords, apple_location, board)
		board = refresh_board
		# 如果蛇可以吃到食物
		if result:
			BESTMOVE = findSafeWay(snake_coords, board, apple_location)
		else:
			BESTMOVE = followTail(snake_coords, board, apple_location)
		if BESTMOVE == ERR:
			BESTMOVE = anyPossibleMove(snake_coords, board, apple_location)
		if BESTMOVE != ERR:
			new_head = findSnakeHead(snake_coords, BESTMOVE)
			snake_coords.insert(0, new_head)
			head_idx = snake_coords[HEADINDEX]['x'] + snake_coords[HEADINDEX]['y'] * MATRIX_W
			end_idx = snake_coords[-1]['x'] + snake_coords[-1]['y'] * MATRIX_W
			if (snake_coords[HEADINDEX]['x'] == apple_location['x']) and (snake_coords[HEADINDEX]['y'] == apple_location['y']):
				board[head_idx] = SNAKENUM
				if len(snake_coords) < MATRIX_SIZE:
					apple_location = GetAppleLocation(snake_coords)
			else:
				board[head_idx] = SNAKENUM
				board[end_idx] = SPACENUM
				del snake_coords[-1]
		else:
			return
		pygame.display.update()
		clock.tick(FPS)


'''主函数'''
def main():
	global screen, default_font, clock
	pygame.init()
	screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	pygame.display.set_caption('AI Snake')
	default_font = pygame.font.Font('simkai.ttf', 18)
	clock = pygame.time.Clock()
	while True:
		RunGame()
		ShowEndInterface()


'''run'''
if __name__ == '__main__':
	main()