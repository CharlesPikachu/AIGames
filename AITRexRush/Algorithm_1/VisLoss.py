# Loss可视化
# 作者: Charles
# 公众号: Charles的皮卡丘
import pickle
from matplotlib import pyplot as plt


with open('./logger/loss.pkl', 'rb') as f:
	loss_dict = pickle.load(f)
times = []
losses = []
for time, loss in loss_dict.items():
	times.append(time)
	losses.append(loss)
plt.title('Loss trend')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.plot(times, losses)
plt.show()