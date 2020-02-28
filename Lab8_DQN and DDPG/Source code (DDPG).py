#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time

import matplotlib.pyplot as plt


# In[48]:


#####################  hyper parameters  ####################
MAX_EPISODES = 10000
MAX_EP_STEPS = 100
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
RENDER = False
ENV_NAME = 'Pendulum-v0'


# In[49]:


###############################  DDPG  ####################################
class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,400)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(300,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x*2
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim,400)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,400)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out1 = nn.Linear(400, 300)
        self.out1.weight.data.normal_(0, 0.1)  # initialization
        self.out2 = nn.Linear(300,1)
        self.out2.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        net = self.out1(net)
        actions_value = self.out2(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach() # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  
        loss_a = -torch.mean(q) 
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_) 
        q_ = self.Critic_target(bs_,a_)  
        q_target = br + GAMMA*q_ 
        q_v = self.Critic_eval(bs,ba)
    
        td_error = self.loss_td(q_target,q_v)

        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


# In[50]:


###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

a_bound = env.action_space.high
ep_reward_collections = []

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2) # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9999    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r
    
        if j == MAX_EP_STEPS-1:
            ep_reward_collections.append(ep_reward)
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)


# In[65]:


##################### plot figure ##################################
epoch = np.arange(0, MAX_EPISODES, 10)
every10ep_reward_collections = [i for step, i in enumerate(ep_reward_collections)if step % 10 == 0]

plt.figure(figsize=(25, 5))
plt.plot(epoch, every10ep_reward_collections)

plt.xlabel('Step')
plt.ylabel('Moving average ep reward')

plt.show() 


# In[66]:


max(every10ep_reward_collections)

