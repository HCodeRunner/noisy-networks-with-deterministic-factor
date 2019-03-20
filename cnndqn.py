import math, random
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F 
import torch.nn.init as init
import time
from base.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import base.tools as tl

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

game_name = "BreakoutNoFrameskip-v4"
save_dir = "data/dqn_bout_one.npz"

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 300000

num_frames = 10000000
batch_size = 32
gamma      = 0.99
updatefrc = 40000

learningRate = 0.0001

replay_initial = 50000
capacity = 900000

arvg_num = 1

env_id = game_name
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)

print(env.observation_space.shape)

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
current_model = tl.CnnDQN(env.observation_space.shape, env.action_space.n, env)
target_model  = tl.CnnDQN(env.observation_space.shape, env.action_space.n, env)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

update_target(current_model, target_model)

losses_all = []
rewards_all = []
for i in range(arvg_num):
    losses = []
    all_rewards = []
    episode_reward = 0
    frame_list = []
    epsilon_list = []
    state = env.reset()
    current_model = tl.CnnDQN(env.observation_space.shape, env.action_space.n, env)

    target_model  = tl.CnnDQN(env.observation_space.shape, env.action_space.n, env)
    
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()
    
    optimizer = optim.Adam(current_model.parameters(), lr=learningRate)
    
    replay_buffer = tl.BaseReplayBuffer(capacity)
    update_target(current_model, target_model)
    for frame_idx in range(1, num_frames + 1):
        # if frame_idx > 50:
        #     env.render()
        #     time.sleep(3600)
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)
    
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
    
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            epsilon_list.append(epsilon)
            all_rewards.append(episode_reward)
            frame_list.append(frame_idx)
            episode_reward = 0

        if len(replay_buffer) > replay_initial:
            loss = tl.compute_td_loss(batch_size, replay_buffer, current_model, target_model, gamma, optimizer)
            losses.append(loss.data[0])

        if frame_idx % updatefrc == 0:
            update_target(current_model, target_model)

        if frame_idx % 2000 == 0:
            print("This is %d frame of 1000000, range %d, reward is %f"%(frame_idx, i, all_rewards[-1]))

    losses_all.append(losses)
    rewards_all.append(all_rewards)

mean_losses,var_losses = tl.StatShrink2D(losses_all)
mean_rewards,var_rewards = tl.StatShrink2D(rewards_all)
#tl.save2D4list2(mean_losses,var_losses,mean_rewards,var_rewards,epsilon_list,epsilon_list,frame_list,save_dir)
tl.save2D4list(mean_losses,var_losses,mean_rewards,var_rewards,save_dir)
