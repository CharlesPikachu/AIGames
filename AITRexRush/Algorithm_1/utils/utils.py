# 作者: Charles
# 公众号: Charles的皮卡丘
# 定义了一些工具函数
import os
import cv2
import math
import pickle
import numpy as np
from io import BytesIO
from PIL import ImageGrab, Image
import base64


'''
Function:
	截屏
Input:
	-driver: selenium.webdriver
	-bbox: for ImageGrab, example: (10, 20, 600, 300).
Return:
	-img: PIL.Image
'''
def get_screenshot(driver=None, bbox=(0, 0, 450, 150)):
	if driver is None:
		img = ImageGrab.grab(bbox=bbox).convert('L')
	else:
		'''
		# 50, 50, 950, 700
		img = driver.get_screenshot_as_png()
		img = Image.open(BytesIO(img)).convert('L')
		img = img.crop(bbox)
		'''
		image_b64 = driver.execute_script("canvasRunner = document.getElementById('runner-canvas'); \
											return canvasRunner.toDataURL().substring(22)")
		img = Image.open(BytesIO(base64.b64decode(image_b64))).convert('L')
		img = img.crop(bbox)
	return img


'''
Function:
	图像预处理(针对截屏)
Input:
	-img(PIL.Image): 待处理的图像
	-size(tuple): 目标图像大小
	-interpolation: 插值方式
	-use_canny: 是否使用Canny算子预处理
Return:
	-img(np.array): 处理之后的图像
'''
def preprocess_img(img, size=(128, 128), interpolation=cv2.INTER_LINEAR, use_canny=False):
	# if not use_canny:
	# 	img = img.split()[0]
	img = np.array(img)
	img = cv2.resize(img, size, interpolation=interpolation)
	if use_canny:
		img = cv2.Canny(img, threshold1=100, threshold2=200)
	return img


'''
Function:
	保存字典数据
Input:
	-data: dict()
	-savepath: 保存路径
	-savename: 保存文件名
'''
def save_dict(data, savepath, savename):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	if os.path.isfile(os.path.join(savepath, savename)):
		mode = 'ab'
	else:
		mode = 'wb'
	with open(os.path.join(savepath, savename), mode) as f:
		pickle.dump(data, f)


'''
Function:
	读取字典数据
Input:
	-datapath: 数据位置
'''
def read_dict(datapath):
	with open(datapath, 'rb') as f:
		return pickle.load(f)


'''
Function:
	Sigmoid函数
Input:
	-x(int): 输入数据
'''
def sigmoid(x):
	return 1.0 / (1 + math.exp(-x))