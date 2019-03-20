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
from base.noisynetws import NoisyDQN, improved_td_loss


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

env_id = "CartPole-v0"
env = gym.make(env_id)

k_start = 0
reward_inf = 0
reward_sup = 100
k_by_reward = lambda reward_x : ((k_end-k_start)*reward_x+(k_start*reward_sup-reward_inf*k_end))/(reward_sup-reward_inf)
kends = [1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3]

current_model = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)
target_model  = NoisyDQN(env.observation_space.shape[0], env.action_space.n, env)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()
    
optimizer = optim.Adam(current_model.parameters(), lr=0.00005)

replay_buffer = tl.BaseReplayBuffer(10000)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)

losses_all = []
rewards_all = []


for k_end in kends:
    k_start = 0
    reward_inf = -20
    reward_sup = 0
    k_by_reward = lambda reward_x : ((k_end-k_start)*reward_x+(k_start*reward_sup-reward_inf*k_end))/(reward_sup-reward_inf)
    
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

        optimizer = optim.Adam(current_model.parameters(), lr=0.00005)

        replay_buffer = tl.BaseReplayBuffer(10000)
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
        
            if len(replay_buffer) > batch_size:
                loss = improved_td_loss(batch_size, replay_buffer, current_model, target_model, gamma, args_k, optimizer)
                losses.append(loss.data[0])
        
            if frame_idx % 1000 == 0:
                update_target(current_model, target_model)

            if frame_idx % 1000 == 0:
                print("This is %d frame of 1000000, k_end %f, reward is %f"%(frame_idx, k_end*2., all_rewards[-1]))

        losses_all.append(losses)
        rewards_all.append(all_rewards)

    str = "data/analysis/CartPolerl050_kend%f"%(k_end*2.)

    mean_losses,var_losses = tl.StatShrink2D(losses_all)
    mean_rewards,var_rewards = tl.StatShrink2D(rewards_all)
    tl.save2D4list(mean_losses,var_losses,mean_rewards,var_rewards,str)






