'''
主函数
作者: Charles
公众号: Charles的皮卡丘
'''
import config
import tensorflow as tf
from nets.qNet import DQN


def main():
	session = tf.InteractiveSession()
	DQN(config.options).train(session)


if __name__ == '__main__':
	main()