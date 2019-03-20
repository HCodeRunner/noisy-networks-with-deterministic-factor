import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np



def compute_td_loss1(batch_size, replay_buffer, current_model, target_model, gamma, opt):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 

    state      = Variable(torch.cuda.FloatTensor(np.float32(state)))
    next_state = Variable(torch.cuda.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.cuda.LongTensor(action))
    reward     = Variable(torch.cuda.FloatTensor(reward))
    done       = Variable(torch.cuda.FloatTensor(np.float32(done)))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)
    action = action.type(torch.cuda.LongTensor)
    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    reward = reward.type(torch.cuda.FloatTensor)
    next_q_value = next_q_value.type(torch.cuda.FloatTensor)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss  = (q_value - expected_q_value.detach()).pow(2)
    loss  = loss.mean()
        
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss


def compute_td_loss(batch_size, buffer, current_model, target_model, gamma, opt):
    state, action, reward, next_state, done = buffer.sample(batch_size)

    state      = Variable(torch.cuda.FloatTensor(np.float32(state)))
    next_state = Variable(torch.cuda.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.cuda.LongTensor(action))
    reward     = Variable(torch.cuda.FloatTensor(reward))
    done       = Variable(torch.cuda.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)
    next_q_state_values = target_model(next_state) 
    action = action.type(torch.cuda.LongTensor)
    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

def improved_td_loss(batch_size, buffer, current_model, target_model, gamma, args_k, opt):
    state, action, reward, next_state, done = buffer.sample(batch_size)

    state      = Variable(torch.cuda.FloatTensor(np.float32(state)))
    next_state = Variable(torch.cuda.FloatTensor(np.float32(next_state)))
    action     = Variable(torch.cuda.LongTensor(action))
    reward     = Variable(torch.cuda.FloatTensor(reward))
    done       = Variable(torch.cuda.FloatTensor(done))

    q_values      = current_model(state)
    next_q_values = target_model(next_state)
    next_q_state_values = target_model(next_state) 

    q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    #sigmaloss = current_model.get_sigmaloss()
    #sigmaloss = sigmaloss.type(torch.cuda.FloatTensor)
    ##sigmaloss   = Variable(torch.cuda.FloatTensor(np.float32(sigmaloss)).unsqueeze(1), volatile=True)
    #sigmaloss = sigmaloss.type(torch.cuda.FloatTensor)
    #sigmaloss = args_k*sigmaloss

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    ##print(sigmaloss.type)
    #sigmaloss = cudaVDecoder(sigmaloss, batch_size)
    
    #loss = loss + sigmaloss

    opt.zero_grad()
    loss.backward()
    opt.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

def cudaVDecoder(dvariables, size):
    for i in range(size+1):
        if isinstance(dvariables, np.ndarray):
            dvariables = dvariables[0]
        else:
            break
    return dvariables


class NoisyLinear(nn.Module):
    """
        It defines the layer noisy, which will receive p-dim x, and return q-dim y
    """

    # in_features: p, out_features: q
    # so: mu and sigma of weight is q*p; of bias is q.
    def __init__(self, in_features, out_features, std_init=0.4):

        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        # Initialization variance 
        self.std_init     = std_init
        # weight_mu and weight_sigma determined the value of the parameter weight of this layer
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        # bias_mu and bias_sigma determined the value of the parameter bias of this layer
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def sigmaloss(self):
        tensorws = torch.abs(self.weight_sigma)
        ws_mean = torch.mean(tensorws)
        tensorbs = torch.abs(self.bias_sigma)
        bs_mean = torch.mean(tensorbs)
        q = self.out_features
        p = self.in_features
        return (ws_mean*q*p + bs_mean*p)/(q*p+q)

    def getsigma(self):
        tw = torch.mean(self.weight_sigma)
        tw = tw.data.cpu().numpy()
        tb = torch.mean(self.bias_sigma)
        tb = tb.data.cpu().numpy()
        return tw, tb

class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, env):
        super(NoisyDQN, self).__init__()
        self.env = env
        self.linear =  nn.Linear(env.observation_space.shape[0], 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, env.action_space.n)

        
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    
    def act(self, state):
        state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

class CnnNoisyDQN(nn.Module):
    """noisy networks dqn for end-to-end rl"""
    def __init__(self, input_shape, num_actions, env):
        super(CnnNoisyDQN, self).__init__()

        self.env = env
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.noisy1 = NoisyLinear(self.feature_size(), 512)
        self.noisy2 = NoisyLinear(512, env.action_space.n)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = x / 255.
        # now x is on cpu, you need to put it on gpu
        x = x.type(torch.cuda.FloatTensor)
        x = self.features(x)
        x = x.view(batch_size, -1)
        
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x


    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state):
        state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action

    def get_sigmaloss(self):
        #t = Variable(torch.cuda.FloatTensor())
        return self.noisy2.sigmaloss()

    def getsigma(self):
        weight_sigma,bias_sigma = self.noisy2.getsigma()
        return weight_sigma, bias_sigma
