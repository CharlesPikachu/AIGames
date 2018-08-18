# AI五子棋
# 作者： Charles
# 公众号：Charles的皮卡丘
from graphics import *


GRID_WIDTH = 40
COLUMN = 15
ROW = 15
ai_list = []
me_list = []
aime_list = []
all_list = []
next_point = [0, 0]
ratio = 1
DEPTH = 3
cut_count = 0
search_count = 0


# 棋型评分表
scoreModel = [(50, (0, 1, 1, 0, 0)),
			  (50, (0, 0, 1, 1, 0)),
			  (200, (1, 1, 0, 1, 0)),
			  (500, (0, 0, 1, 1, 1)),
			  (500, (1, 1, 1, 0, 0)),
			  (5000, (0, 1, 1, 1, 0)),
			  (5000, (0, 1, 0, 1, 1, 0)),
			  (5000, (0, 1, 1, 0, 1, 0)),
			  (5000, (1, 1, 1, 0, 1)),
			  (5000, (1, 1, 0, 1, 1)),
			  (5000, (1, 0, 1, 1, 1)),
			  (5000, (1, 1, 1, 1, 0)),
			  (5000, (0, 1, 1, 1, 1)),
			  (50000, (0, 1, 1, 1, 1, 0)),
			  (99999999, (1, 1, 1, 1, 1))]


# 判断游戏是否结束
# 四种情况
def is_GameOver(list_now):
	for c in range(COLUMN):
		for r in range(ROW):
			if r < ROW - 4 and (r, c) in list_now and (r+1, c) in list_now and (r+2, c) in list_now and (r+3, c) in list_now and (r+4, c) in list_now:
				return True
			elif c < COLUMN - 4 and (r, c) in list_now and (r, c+1) in list_now and (r, c+2) in list_now and (r, c+3) in list_now and (r, c+4) in list_now:
				return True
			elif r < ROW - 4 and c < COLUMN - 4 and (r, c) in list_now and (r+1, c+1) in list_now and (r+2, c+2) in list_now and (r+3, c+3) in list_now and (r+4, c+4) in list_now:
				return True
			elif r > 3 and c < COLUMN - 4 and (r, c) in list_now and (r-1, c+1) in list_now and (r-2, c+2) in list_now and (r-3, c+3) in list_now and (r-4, c+4) in list_now:
				return True
	return False


# 计算每个方向上的分值
# list1是下子方
# scores_all用于避免重复计算和奖励棋型相交
def calc_score(r, c, x_direction, y_direction, list1, list2, scores_all):
	add_score = 0
	max_score = (0, None)
	# 避免重复计算
	for score_all in scores_all:
		for ps in score_all[1]:
			if r == ps[0] and c == ps[1] and x_direction == score_all[2][0] and y_direction == score_all[2][1]:
				return 0, scores_all
	# 获得棋型
	for noffset in range(-5, 1):
		position = []
		for poffset in range(0, 6):
			x, y = r + (poffset + noffset) * x_direction, c + (poffset + noffset) * y_direction
			if (x, y) in list2:
				position.append(2)
			elif (x, y) in list1:
				position.append(1)
			else:
				position.append(0)
		temp_shape5 = tuple([i for i in position[0: -1]])
		temp_shape6 = tuple(position)
		for score, shape in scoreModel:
			if temp_shape5 == shape or temp_shape6 == shape:
				if score > max_score[0]:
					max_score = (score, ((r + (0 + noffset) * x_direction, c + (0 + noffset) * y_direction),
										 (r + (1 + noffset) * x_direction, c + (1 + noffset) * y_direction),
										 (r + (2 + noffset) * x_direction, c + (2 + noffset) * y_direction),
										 (r + (3 + noffset) * x_direction, c + (3 + noffset) * y_direction),
										 (r + (4 + noffset) * x_direction, c + (4 + noffset) * y_direction)), (x_direction, y_direction))
	# 如果棋型相交，则得分增加
	if max_score[1] is not None:
		for score_all in scores_all:
			for ps1 in score_all[1]:
				for ps2 in max_score[1]:
					if ps1 == ps2 and max_score[0] > 10 and score_all[0] > 10:
						add_score += max_score[0] + score_all[0]
		scores_all.append(max_score)
	return add_score + max_score[0], scores_all


# 评估函数
def evaluation(is_ai):
	total_score = 0
	if is_ai:
		list1 = ai_list
		list2 = me_list
	else:
		list1 = me_list
		list2 = ai_list
	# 评估主动方(下子方)
	Active_scores_all = []
	Active_score = 0
	for l1 in list1:
		r, c = l1[0], l1[1]
		temp_score, Active_scores_all = calc_score(r, c, 0, 1, list1, list2, Active_scores_all)
		Active_score += temp_score
		temp_score, Active_scores_all = calc_score(r, c, 1, 0, list1, list2, Active_scores_all)
		Active_score += temp_score
		temp_score, Active_scores_all = calc_score(r, c, 1, 1, list1, list2, Active_scores_all)
		Active_score += temp_score
		temp_score, Active_scores_all = calc_score(r, c, -1, 1, list1, list2, Active_scores_all)
		Active_score += temp_score
	# 评估被动方(非下子方)
	Passive_scores_all = []
	Passive_score = 0
	for l2 in list2:
		r, c = l2[0], l2[1]
		temp_score, Passive_scores_all = calc_score(r, c, 0, 1, list2, list1, Passive_scores_all)
		Passive_score += temp_score
		temp_score, Passive_scores_all = calc_score(r, c, 1, 0, list2, list1, Passive_scores_all)
		Passive_score += temp_score
		temp_score, Passive_scores_all = calc_score(r, c, 1, 1, list2, list1, Passive_scores_all)
		Passive_score += temp_score
		temp_score, Passive_scores_all = calc_score(r, c, -1, 1, list2, list1, Passive_scores_all)
		Passive_score += temp_score
	# 总评
	total_score = Active_score - Passive_score * ratio * 0.1
	return total_score


# 重新排列未落子的位置列表
# 假设离最后落子的邻居位置最有可能是最优点
def Rearrange(blank_list):
	last_step = aime_list[-1]
	for bl in blank_list:
		for i in range(-1, 2):
			for j in range(-1, 2):
				if i == 0 and j == 0:
					continue
				next_step = (last_step[0]+i, last_step[1]+j)
				if next_step in blank_list:
					blank_list.remove(next_step)
					blank_list.insert(0, next_step)
	return blank_list


# 判断下一步位置是否存在相邻的子
def has_neighbor(next_step):
	for i in range(-1, 2):
		for j in range(-1, 2):
			if i == 0 and j == 0:
				continue
			if (next_step[0]+i, next_step[1]+j) in aime_list:
				return True
	return False


# 负极大值搜索  alpha+beta剪枝
# is_ai： AI方下还是我方下
def negativeMax(is_ai, depth, alpha, beta):
	if is_GameOver(ai_list) or is_GameOver(me_list) or depth == 0:
		return evaluation(is_ai)
	# 未落子的位置
	blank_list = list(set(all_list).difference(set(aime_list)))
	blank_list = Rearrange(blank_list)
	for next_step in blank_list:
		global search_count
		search_count += 1
		if not has_neighbor(next_step):
			continue
		if is_ai:
			ai_list.append(next_step)
		else:
			me_list.append(next_step)
		aime_list.append(next_step)
		value = -negativeMax(not is_ai, depth-1, -beta, -alpha)
		if is_ai:
			ai_list.remove(next_step)
		else:
			me_list.remove(next_step)
		aime_list.remove(next_step)
		if value > alpha:
			if depth == DEPTH:
				next_point[0], next_point[1] = next_step[0], next_step[1]
			if value >= beta:
				global cut_count
				cut_count += 1
				return beta
			alpha = value
	return alpha


# AI下棋
def AI():
	global cut_count
	global search_count
	# 剪枝次数
	cut_count = 0
	# 搜索次数
	search_count = 0
	negativeMax(True, DEPTH, -99999999, 99999999)
	print('[Cut_Count]: %d, [Search_Count]: %d' % (cut_count, search_count))
	return next_point[0], next_point[1]


# 画棋盘
def GobangWin():
	gw = GraphWin('AI Gobang', GRID_WIDTH*COLUMN, GRID_WIDTH*ROW)
	gw.setBackground('gray')
	for j in range(0, GRID_WIDTH*COLUMN+1, GRID_WIDTH):
		l = Line(Point(j, 0), Point(j, GRID_WIDTH*COLUMN))
		l.draw(gw)
	for i in range(0, GRID_WIDTH*ROW+1, GRID_WIDTH):
		l = Line(Point(0, i), Point(GRID_WIDTH*ROW, i))
		l.draw(gw)
	return gw


# 主程序
def run():
	# 初始化
	gw = GobangWin()
	for j in range(COLUMN+1):
		for i in range(ROW+1):
			all_list.append((i, j))
	# 游戏是否结束flag
	is_game = True
	# 统计步数，用于判断现在轮到谁落子，奇数为AI方，偶数为我方
	step_count = 0
	while is_game:
		if step_count % 2:
			p_ai = AI()
			if p_ai in aime_list:
				message = Text(Point(300, 300), 'AI gets a wrong next step.')
				message.draw(gw)
				is_game = False
			ai_list.append(p_ai)
			aime_list.append(p_ai)
			piece = Circle(Point(GRID_WIDTH * p_ai[0], GRID_WIDTH * p_ai[1]), 16)
			piece.setFill('white')
			piece.draw(gw)
			if is_GameOver(ai_list):
				message = Text(Point(100, 100), 'AI white win.')
				message.draw(gw)
				is_game = False
			step_count += 1
		else:
			p_me = gw.getMouse()
			x = round((p_me.getX()) / GRID_WIDTH)
			y = round((p_me.getY()) / GRID_WIDTH)
			if not ((x, y) in aime_list):
				me_list.append((x, y))
				aime_list.append((x, y))
				piece = Circle(Point(GRID_WIDTH * x, GRID_WIDTH * y), 16)
				piece.setFill('black')
				piece.draw(gw)
				if is_GameOver(me_list):
					message = Text(Point(100, 100), 'You black win.')
					message.draw(gw)
					is_game = False
				step_count += 1
	# 游戏结束后的处理
	message = Text(Point(300, 300), 'Click anywhere to quit.')
	message.draw(gw)
	gw.getMouse()
	gw.close()





if __name__ == '__main__':
	run()