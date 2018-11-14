'''
Function:
	AI玩俄罗斯方块
Author:
	Charles
公众号:
	Charles的皮卡丘
'''
import copy
import math
from modules.utils import *


'''
Function:
	AI玩俄罗斯方块
'''
class TetrisAI():
	def __init__(self, inner_board):
		self.inner_board = inner_board
	'''获得下一步的行动'''
	def getNextAction(self):
		if self.inner_board.current_tetris == tetrisShape().shape_empty:
			return None
		action = None
		# 当前可操作的俄罗斯方块的direction范围
		if self.inner_board.current_tetris.shape in [tetrisShape().shape_O]:
			current_direction_range = [0]
		elif self.inner_board.current_tetris.shape in [tetrisShape().shape_I, tetrisShape().shape_Z, tetrisShape().shape_S]:
			current_direction_range = [0, 1]
		else:
			current_direction_range = [0, 1, 2, 3]
		# 下一个可操作的俄罗斯方块的direction范围
		if self.inner_board.next_tetris.shape in [tetrisShape().shape_O]:
			next_direction_range = [0]
		elif self.inner_board.next_tetris.shape in [tetrisShape().shape_I, tetrisShape().shape_Z, tetrisShape().shape_S]:
			next_direction_range = [0, 1]
		else:
			next_direction_range = [0, 1, 2, 3]
		# 简单的AI算法
		for d_now in current_direction_range:
			x_now_min, x_now_max, y_now_min, y_now_max = self.inner_board.current_tetris.getRelativeBoundary(d_now)
			for x_now in range(-x_now_min, self.inner_board.width - x_now_max):
				board = self.getFinalBoardData(d_now, x_now)
				for d_next in next_direction_range:
					x_next_min, x_next_max, y_next_min, y_next_max = self.inner_board.next_tetris.getRelativeBoundary(d_next)
					distances = self.getDropDistances(board, d_next, range(-x_next_min, self.inner_board.width-x_next_max))
					for x_next in range(-x_next_min, self.inner_board.width-x_next_max):
						score = self.calcScore(copy.deepcopy(board), d_next, x_next, distances)
						if not action or action[2] < score:
							action = [d_now, x_now, score]
		return action
	'''获取当前俄罗斯方块在某位置以某方向下落到最低点时的板块数据'''
	def getFinalBoardData(self, d_now, x_now):
		board = copy.deepcopy(self.inner_board.getBoardData())
		dy = self.inner_board.height - 1
		for x, y in self.inner_board.current_tetris.getAbsoluteCoords(d_now, x_now, 0):
			count = 0
			while (count + y < self.inner_board.height) and (count + y < 0 or board[x + (count + y) * self.inner_board.width] == tetrisShape().shape_empty):
				count += 1
			count -= 1
			if dy > count:
				dy = count
		return self.imitateDropDown(board, self.inner_board.current_tetris, d_now, x_now, dy)
	'''模拟下落到最低点获得板块数据'''
	def imitateDropDown(self, board, tetris, direction, x_imitate, dy):
		for x, y in tetris.getAbsoluteCoords(direction, x_imitate, 0):
			board[x + (y + dy) * self.inner_board.width] = tetris.shape
		return board
	'''获取下一个俄罗斯方块x_range范围内某状态俄罗斯方块到最低点的距离'''
	def getDropDistances(self, board, direction, x_range):
		dists = {}
		for x_next in x_range:
			if x_next not in dists:
				dists[x_next] = self.inner_board.height - 1
			for x, y in self.inner_board.next_tetris.getAbsoluteCoords(direction, x_next, 0):
				count = 0
				while (count + y < self.inner_board.height) and (count + y < 0 or board[x + (count + y) * self.inner_board.width] == tetrisShape().shape_empty):
					count += 1
				count -= 1
				if dists[x_next] > count:
					dists[x_next] = count
		return dists
	'''计算某方案的得分'''
	def calcScore(self, board, d_next, x_next, distances):
		# 下个俄罗斯方块以某种方式模拟到达底部
		board = self.imitateDropDown(board, self.inner_board.next_tetris, d_next, x_next, distances[x_next])
		width, height = self.inner_board.width, self.inner_board.height
		# 下一个俄罗斯方块以某方案行动到达底部后的得分(可消除的行数)
		removed_lines = 0
		# 空位统计
		hole_statistic_0 = [0] * width
		hole_statistic_1 = [0] * width
		# 方块数量
		num_blocks = 0
		# 空位数量
		num_holes = 0
		# 每个x位置堆积俄罗斯方块的最高点
		roof_y = [0] * width
		for y in range(height-1, -1, -1):
			# 是否有空位
			has_hole = False
			# 是否有方块
			has_block = False
			for x in range(width):
				if board[x + y * width] == tetrisShape().shape_empty:
					has_hole = True
					hole_statistic_0[x] += 1
				else:
					has_block = True
					roof_y[x] = height - y
					if hole_statistic_0[x] > 0:
						hole_statistic_1[x] += hole_statistic_0[x]
						hole_statistic_0[x] = 0
					if hole_statistic_1[x] > 0:
						num_blocks += 1
			if not has_block:
				break
			if not has_hole and has_block:
				removed_lines += 1
		# 数据^0.7之和
		num_holes = sum([i ** .7 for i in hole_statistic_1])
		# 最高点
		max_height = max(roof_y) - removed_lines
		# roof_y做差分运算
		roof_dy = [roof_y[i]-roof_y[i+1] for i in range(len(roof_y)-1)]
		# 计算标准差E(x^2) - E(x)^2
		if len(roof_y) <= 0:
			roof_y_std = 0
		else:
			roof_y_std = math.sqrt(sum([y**2 for y in roof_y]) / len(roof_y) - (sum(roof_y) / len(roof_y)) ** 2)
		if len(roof_dy) <= 0:
			roof_dy_std = 0
		else:
			roof_dy_std = math.sqrt(sum([dy**2 for dy in roof_dy]) / len(roof_dy) - (sum(roof_dy) / len(roof_dy)) ** 2)
		# roof_dy绝对值之和
		abs_dy = sum([abs(dy) for dy in roof_dy])
		# 最大值与最小值之差
		max_dy = max(roof_y) - min(roof_y)
		# 计算得分
		score = removed_lines * 1.8 - num_holes * 1.0 - num_blocks * 0.5 - max_height ** 1.5 * 0.02 - roof_y_std * 1e-5 - roof_dy_std * 0.01 - abs_dy * 0.2 - max_dy * 0.3
		return score