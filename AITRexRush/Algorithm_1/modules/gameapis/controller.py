'''
Function:
	Control the T-Rex in chrome browser
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import time
import base64
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options


'''game controller'''
class GameController():
	def __init__(self, cfg, **kwargs):
		chrome_options = Options()
		chrome_options.add_argument("disable-infobars")
		self.driver = webdriver.Chrome(
										executable_path=cfg.DRIVER_PATH,
										chrome_options=chrome_options
									)
		self.driver.maximize_window()
		self.driver.get(cfg.GAME_URL)
		self.driver.execute_script("Runner.config.ACCELERATION=0")
		self.driver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")
		self.jump()
	'''run'''
	def run(self, action):
		assert len(action) == 2
		# operate T-Rex according to the action
		if action[1] > action[0]:
			self.jump()
		# get score
		score = self.state('score')
		# whether die or not
		if self.state('crashed'):
			self.restart()
			is_dead = True
		else:
			is_dead = False
		# get game image
		image = self.screenshot()
		# return necessary info
		return image, score, is_dead
	'''get the game state'''
	def state(self, type_):
		assert type_ in ['crashed', 'playing', 'score']
		if type_ == 'crashed':
			return self.driver.execute_script("return Runner.instance_.crashed")
		elif type_ == 'playing':
			return self.driver.execute_script("return Runner.instance_.playing")
		else:
			digits = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
			score = ''.join(digits)
			return int(score)
	'''jump'''
	def jump(self):
		self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
	'''bow the head'''
	def bowhead(self):
		self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
	'''pause the game'''
	def pause(self):
		return self.driver.execute_script("return Runner.instance_.stop()")
	'''restart the game'''
	def restart(self):
		self.driver.execute_script("Runner.instance_.restart()")
		time.sleep(0.2)
	'''resume the game'''
	def resume(self):
		return self.driver.execute_script("return Runner.instance_.play()")
	'''stop the game'''
	def stop(self):
		self.driver.close()
	'''screenshot'''
	def screenshot(self, area=(0, 0, 450, 150)):
		image_b64 = self.driver.execute_script("canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)")
		image = Image.open(BytesIO(base64.b64decode(image_b64))).convert('L')
		image = image.crop(area)
		return image