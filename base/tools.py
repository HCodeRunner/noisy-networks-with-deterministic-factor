import math, random
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init

from collections import deque

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class BaseReplayBuffer(object):
	"""native replay buffer in dqn2015"""
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		# reshap a n-dim array as a 1*n-dim matrix
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)
		# put a 5 tuple into replay buffer
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		# random.sample can get array of tuple, zip* can get array of array
		state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
		# concatenate batch_size * (1*n) as (batch_size*n) 
		return np.concatenate(state), action, reward, np.concatenate(next_state),done

	def __len__(self):
		return len(self.buffer)

# class baseDQN(nn.Module):
# 	def __init__(self, input_shape, num_actions, env):
# 		super(baseDQN, self).__init__()
# 		self.env = env
# 		self.layers = nn.Sequential(
# 			nn.Linear(env.observation_space.shape[0], 128),
# 			nn.ReLU(),
# 			nn.Linear(128, 128),
# 			nn.ReLU(),
# 			nn.Linear(128, env.action_space.n)
# 		)

# 	def forward(self, x):
# 		return self.layers(x)

# 	def act(self, state, epsilon):
# 		if random.random()>epsilon:
# 			state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
# 			q_value = self.forward(state)
# 			action = q_value.max(1)[1].data[0]
# 		else:
# 			action = random.randrange(self.env.action_space.n)
# 		return action

class baseDQN(nn.Module):
    def __init__(self, input_shape, num_actions, env):
        super(baseDQN, self).__init__()
        self.env = env
        self.linear1 = nn.Linear(env.observation_space.shape[0], 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return self.linear3(x)

    def act(self, state, epsilon):
        if random.random()>epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def reinit(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight.data)
                init.constant(m.bias.data,0.1)

# class CnnDQN(nn.Module):
#     def __init__(self, input_shape, num_actions,env):
#         super(CnnDQN, self).__init__()
#         self.env = env

#         self.input_shape = input_shape
#         self.num_actions = num_actions

#         self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.relu3 = nn.ReLU()
#         self.linear1 = nn.Linear(self.conv3(), 512)
#         self.relu4 = nn.ReLU()
#         self.linear2 = nn.Linear(512, self.num_actions)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear1(x)
#         x = self.relu4(x)
#         x = self.linear2(x)
#         return x 

#     def feature_size(self):
#         return self.conv3(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

#     def act(self, state, epsilon):
#         if random.random() > epsilon:
#             state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
#             q_value = self.forward(state)
#             action  = q_value.max(1)[1].data[0]
#         else:
#             action = random.randrange(self.env.action_space.n)
#         return action

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions,env):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.env = env

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action


def compute_td_loss(batch_size, buffer, current_model, target_model, gamma, opt):
    state , action, reward, next_state, done = buffer.sample(batch_size)
    #print(sys.getsizeof(state))

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma*next_q_value*(1-done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss


def Shrink2D(data_list):
    """
      args:
        A two-dimensional list of different lengths per line
      Function：
        Shrink to a two-dimensional list which has the same length of each row with the minimum row length
      return:
        a two-dimensional list
    """
    assert isinstance(data_list, list),"params is not list"
    assert isinstance(data_list[0], list),"params is not list2d"
    temp_data = data_list[0]
    for x in range(data_list.__len__()-1):
        list2 = data_list[x+1]
        temp_data = [temp_data[index]+list2[index] for index in range(min(temp_data.__len__(),list2.__len__()))]
    return temp_data

def transpose(matrix_list):
    """
      2D list transpose, requiring the same length per row
    """
    return [[row[col] for row in matrix_list] for col in range(len(matrix_list[0]))]

def StatShrink2D(data_list):
    """
      args:
        A two-dimensional list of different lengths per line
      Function：
        based on avg_reward Calculate the mean, variance of each col
      return:
        a mean list,a var list
    """
    assert isinstance(data_list, list),"params is not list"
    assert isinstance(data_list[0], list),"params is not list2d"
    len_data = [x.__len__() for x in data_list]
    min_len = min(len_data)
    new_list = []
    for ldata in data_list:
        nlist = [ldata[index] for index in range(min_len)]
        new_list.append(nlist)
    new_list = transpose(new_list)
    mean_list = [np.mean(mdata) for mdata in new_list]
    var_list = [np.var(edata) for edata in new_list]
    return mean_list,var_list

def dropmean(mdata):
    mdata.sort()
    print(mdata.size)
    return np.mean(mdata[1:-1])

def dropvar(edata):
    edata.sort()
    return np.var(edata[1:-1])


def save2D4list(a, b, c, d, dirs):
    np.savez(dirs, a, b, c, d)

def save2D4list2(a, b, c, d, e, f, g, dirs):
    np.savez(dirs, a, b, c, d, e, f, g)

def load2D4list(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"]

def load2D4list2(dirs):
    r = np.load(dirs)
    return r["arr_0"],r["arr_1"],r["arr_2"],r["arr_3"],r["arr_4"],r["arr_5"],r["arr_6"]

def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()