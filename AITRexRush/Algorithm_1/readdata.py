import pickle


f = open('E:\DQN__T-Rex Rush\data_1\Data_deque_0.pkl', 'rb')
data = pickle.load(f)
print(len(data))