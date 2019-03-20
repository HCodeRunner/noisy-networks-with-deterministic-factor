import math, random
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F 
import torch.nn.init as init
import matplotlib.pyplot as plt
import base.tools as tl
from base.noisynetws import NoisyDQN, compute_td_loss1


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_id = "CartPole-v0"
env = gym.make(env_id)


current_model = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)
target_model  = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_buffer = tl.BaseReplayBuffer(10000)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)

losses_all = []
rewards_all = []

for i in range(5):
    num_frames = 30000
    batch_size = 32
    gamma      = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0
    current_model = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)
    target_model  = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

    state = env.reset()

    optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

    replay_buffer = tl.BaseReplayBuffer(10000)
    update_target(current_model, target_model)

    for frame_idx in range(1, num_frames + 1):
        action = current_model.act(state)
    
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
    
        state = next_state
        episode_reward += reward
    
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        
        if len(replay_buffer) > batch_size:
            loss = compute_td_loss1(batch_size, replay_buffer, current_model, target_model, gamma, optimizer)
            losses.append(loss.data[0])
        
        if frame_idx % 1000 == 0:
            update_target(current_model, target_model)

        if frame_idx % 1000 == 0:
            print("This is %d frame of 1000000, range %d, reward is %f"%(frame_idx, 1, all_rewards[-1]))

    losses_all.append(losses)
    rewards_all.append(all_rewards)

mean_losses,var_losses = tl.StatShrink2D(losses_all)
mean_rewards,var_rewards = tl.StatShrink2D(rewards_all)
tl.save2D4list(mean_losses,var_losses,mean_rewards,var_rewards,"data/noisydqn_CartPole_five1.npz")