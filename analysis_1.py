import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F 
import torch.nn.init as init
import torch.autograd as autograd
from base.wrappers import make_atari, wrap_deepmind, wrap_pytorch

import base.tools as tl
from base.noisynetws import improved_td_loss, CnnNoisyDQN

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

num_frames = 1000000
batch_size = 32
gamma      = 0.99
updatefrc = 1000

learningRate = 0.0001

replay_initial = 10000
capacity = 100000

game_name = "PongNoFrameskip-v4"
save_dir = "data/analysis/rl1_"
kends = [1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3]

env_id = game_name
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

current_model = CnnNoisyDQN(env.observation_space.shape, env.action_space.n, env)
target_model  = CnnNoisyDQN(env.observation_space.shape, env.action_space.n, env)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
optimizer = optim.Adam(current_model.parameters(), lr = learningRate)

replay_buffer = tl.BaseReplayBuffer(capacity)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)

for k_end in kends:
    k_start = 0
    reward_inf = -20
    reward_sup = 0
    k_by_reward = lambda reward_x : ((k_end-k_start)*reward_x+(k_start*reward_sup-reward_inf*k_end))/(reward_sup-reward_inf)
    
    losses = []
    all_rewards = []

    episode_reward = 0
    state = env.reset()
    current_model = CnnNoisyDQN(env.observation_space.shape, env.action_space.n, env)
    target_model  = CnnNoisyDQN(env.observation_space.shape, env.action_space.n, env)
    
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    optimizer = optim.Adam(current_model.parameters(), lr = learningRate)

    replay_buffer = tl.BaseReplayBuffer(capacity)
    update_target(current_model, target_model)
    for frame_idx in range(1, num_frames + 1):
        args_k = 0.
        action = current_model.act(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            args_k = k_by_reward(episode_reward)
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0


        if len(replay_buffer) > replay_initial:
            loss = improved_td_loss(batch_size, replay_buffer, current_model, target_model, gamma, args_k, optimizer)
            losses.append(loss.data[0])

        if frame_idx % updatefrc == 0:
            update_target(current_model, target_model)

        if frame_idx % 2000 == 0:
            print("This is %d frame of 1000000, k_end %d, reward is %f"%(frame_idx, k_end, all_rewards[-1]))
    
    save_dir = "data/analysis/rl1_kend%d"%(k_end*2)
    tl.save2D4list(losses, [], all_rewards, [], save_dir)

















