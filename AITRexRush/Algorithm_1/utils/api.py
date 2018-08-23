# 作者: Charles
# 公众号: Charles的皮卡丘
# python控制恐龙行动的API接口
import os
import sys
import time
import numpy as np
from PIL import ImageOps
from PIL import ImageGrab
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
if __name__ == '__main__':
	from utils import *
else:
	sys.path.append('..')
	from utils.utils import *


'''
Function:
	接口一:利用selenium使python和浏览器交互
	直接运行即可
'''
class API_Selenium():
	def __init__(self, game_url, custom_config=True):
		chrome_options = Options()
		chrome_options.add_argument("disable-infobars")
		if __name__ == '__main__':
			executable_path = os.path.join(os.getcwd()[0:-5], 'driver/chromedriver.exe')
		else:
			executable_path = os.path.join(os.getcwd(), 'driver/chromedriver.exe')
		self.driver = webdriver.Chrome(
										executable_path=executable_path,
										chrome_options=chrome_options
										)
		self.driver.maximize_window()
		# self.driver.set_window_position(x=-10, y=0)
		self.driver.get(game_url)
		# 一些基本设置
		if custom_config:
			self.driver.execute_script("Runner.config.ACCELERATION=0")
			self.driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
	# 恐龙跳跃
	def jump(self):
		self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
	# 恐龙低头
	def down(self):
		self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
	# 获得游戏当前的状态
	def get_state(self, opt):
		if opt == 'crashed':
			return self.driver.execute_script("return Runner.instance_.crashed")
		elif opt == 'playing':
			return self.driver.execute_script("return Runner.instance_.playing")
		elif opt == 'score':
			score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
			score = ''.join(score_array)
			return int(score)
		else:
			return None
	# 重新开始游戏
	def restart(self):
		self.driver.execute_script("Runner.instance_.restart()")
		time.sleep(0.1)
	# 游戏暂停
	def pause(self):
		return self.driver.execute_script("return Runner.instance_.stop()")
	# 游戏继续
	def resume(self):
		return self.driver.execute_script("return Runner.instance_.play()")
	# 结束游戏
	def stop(self):
		self.driver.close()


'''
Function:
	接口二: 利用pyautogui使python与浏览器交互
	需要手动打开游戏界面(界面最大化)
'''
class API_Pyautogui():
	def __init__(self, replay_btn=(341, 415)):
		self.replay_btn = replay_btn
	# 跳跃
	def jump(self):
		pyautogui.keyDown('space')
		time.sleep(0.05)
		pyautogui.keyUp('space')
	# 低头
	def down(self, time_interval=0.1):
		pyautogui.keyDown('down')
		time.sleep(time_interval)
		pyautogui.keyUp('down')
	# 重新开始游戏
	def restart(self):
		pyautogui.click(self.replay_btn)
		self.jump()
		time.sleep(0.1)
	# 获得游戏当前的状态
	def get_state(self, bbox=None):
		img = ImageGrab.grab(bbox=bbox)
		grayImg = ImageOps.grayscale(img)
		temp = np.array(grayImg.getcolors())
		return temp.sum()


'''
Function:
	恐龙的Agent(神经网络控制)
Input:
	-api: 选择接口类型
'''
class DinoAgent():
	def __init__(self, game_url, api='selenium'):
		if api == 'selenium':
			self.gameAPI = API_Selenium(game_url)
		elif api == 'pyautogui':
			self.gameAPI = API_Pyautogui()
		else:
			raise ValueError('DinoAgent-api param error...')
		self.api = api
		self.gameAPI.jump()
		time.sleep(2)
	# 通过完成一次动作获取训练所需的所有数据
	# 一般只需要调用这个就行了
	def perform_operation(self, actions):
		assert len(actions) == 2
		score = self.get_score()
		is_dead = False
		if actions[1] > actions[0]:
			self.jump()
		if self.is_crashed():
			self.restart()
			is_dead = True
			# time.sleep(0.5)
		img = self.screenshot()
		return img, score, is_dead
	# 跳跃
	def jump(self):
		self.gameAPI.jump()
	# 低头
	def down(self):
		self.gameAPI.down()
	# 重新开始
	def restart(self):
		self.gameAPI.restart()
	# 恐龙是否在奔跑
	def is_running(self):
		if self.api == 'selenium':
			return self.gameAPI.get_state(opt='playing')
	# 恐龙是否GG了
	def is_crashed(self):
		if self.api == 'selenium':
			return self.gameAPI.get_state(opt='crashed')
	# 获得分数
	def get_score(self):
		if self.api == 'selenium':
			return self.gameAPI.get_state(opt='score')
	# 暂停
	def pause(self):
		self.gameAPI.pause()
	# 重新开始
	def resume(self):
		self.gameAPI.resume()
	# 截屏
	def screenshot(self):
		if self.api == 'selenium':
			img = get_screenshot(driver=self.gameAPI.driver)
		else:
			img = get_screenshot()
		return img


# 调试用
if __name__ == '__main__':
	da = DinoAgent()
	time.sleep(2)
	img = da.screenshot()
	img.show()