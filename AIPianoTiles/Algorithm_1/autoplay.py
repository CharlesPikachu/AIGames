'''
Function:
	http://www.4399.com/flash/154247_3.htm自动玩这个游戏
作者:
	Charles
公众号:
	Charles的皮卡丘
'''
import os
import cv2
import pyautogui
import numpy as np
import pyscreenshot
# from selenium import webdriver


'''
Function:
	与浏览器交互类
'''
class API_Class():
	def __init__(self):
		self.game_url = 'http://www.4399.com/flash/154247_3.htm'
		# self.driver = webdriver.Chrome(executable_path='driver/chromedriver.exe')
		# self.driver.maximize_window()
		# self.driver.get(self.game_url)
	# 鼠标点击
	def click(self, position):
		# pyautogui.moveTo(position[0], position[1], duration=1)
		pyautogui.click(position[0], position[1])
	# 截屏
	def screenshot(self, bbox=(566, 220, 968, 823)):
		return pyscreenshot.grab(bbox=bbox)


'''
Function:
	自动玩“别再踩白块了”
'''
def auto_play(bbox):
	api = API_Class()
	count = 0
	click_num = 100
	while True:
		img_rgb = api.screenshot(bbox=bbox)
		img_rgb = np.array(img_rgb)
		gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
		ret, gray_thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
		# 形态学处理-腐蚀
		gray_erode = cv2.erode(gray_thresh, None, iterations=5)
		# 轮廓检测
		img, contours, hierarchy = cv2.findContours(gray_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for contour in contours:
			min_1 = tuple(contour[contour[:, :, 1].argmin()][0])
			min_2 = tuple(contour[contour[:, :, 0].argmin()][0])
			max_1 = tuple(contour[contour[:, :, 1].argmax()][0])
			max_2 = tuple(contour[contour[:, :, 0].argmax()][0])
			# 如果检测到的是小轮廓，应该不是需要点击的黑块
			if max_1[1] - min_1[1] < 50:
				continue
			x = (min_2[0] + max_2[0]) // 2
			y = max_1[1] - 15
			position = (x + bbox[0], y + bbox[1])
			api.click(position)
			count += 1
		if count > click_num:
			break


if __name__ == '__main__':
	bbox = (566, 220, 968, 823)
	auto_play(bbox)