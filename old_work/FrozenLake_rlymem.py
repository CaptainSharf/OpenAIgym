import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.optim as optim
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.l1 = nn.Linear(16,30)
        self.l2 = nn.Linear(30,4)
        self.crit = nn.MSELoss()
    def forward(self, x):
        x = F.relu(self.l1(x))
        x =  F.relu(self.l2(x))
        return x

def evaluate(env,net):
    num_episodes = 100
    num_steps = []
    reward = []
    t_reward = 0
    sum_j = 0
    for i in range(1,num_episodes+1):
        s = env.reset()
        d = False
        j = 0
        while j<99:
            inp = np.identity(16)[s:s+1]
            inp = Variable(torch.Tensor(inp))
            out  = net.forward(inp)
            Q_val,act = torch.max(out,1)
            Q_val = Q_val.data.numpy()
            Q_val = Q_val[0][0]
            act  = act.data.numpy()
            #print j
            act = act[0][0]
            s,r,d,_ = env.step(act)
            t_reward+=r
            if d==True:
                break
            j+=1
        if i%10==0:
            sum_j+=j
            #print t_reward
            reward.append(t_reward/10)
            num_steps.append(sum_j/10)
            t_reward = 0
            sum_j = 0

    plt.plot(reward)
    plt.xlabel('num_episodes')
    plt.ylabel('Average reward')
    plt.savefig('rewards_rply.png')

    plt.plot(num_steps)
    plt.xlabel('num_episodes')
    plt.ylabel('avg_num_timesteps')
    plt.savefig('steps_rply.png')


gamma = 0.99
#Instantiated Neural Net
net = model()


optimizer = optim.SGD(net.parameters(),lr=0.1)
num_steps = []
reward = []
num_episodes = 100
epsilon = 1

for i in range(1,num_episodes+1):
    s = env.reset()
    d = False
    epsilon = 0.1
    replay_memory = []
    j=0
    while j<99:
        inp = np.identity(16)[s:s+1]
        inp = Variable(torch.Tensor(inp))
        out  = net.forward(inp)
        Q_val,act = torch.max(out,1)
        Q_val = Q_val.data.numpy()
        Q_val = Q_val[0][0]
        act  = act.data.numpy()
        act = act[0][0]

        if np.random.random(1) < epsilon:
            act = env.action_space.sample()
        s1,r,d,_ = env.step(act)

        #t_reward+=r
        inp1 = np.identity(16)[s1:s1+1]
        inp1 = Variable(torch.Tensor(inp1))
        out  = net.forward(inp1)
        Q_new,_ = torch.max(out,1)
        Q_new = Q_new.data.numpy()
        Q_new = Q_new[0][0]
        val  = r+ gamma*Q_new
        replay_memory.append([Q_val,val])
        s=s1
        j+=1
        if d==True:
            break
            epsilon = 1./((i/50) + 10)

    #sum_j+=j #How long did the episode last

    output = np.array(replay_memory[:][0])
    target = np.array(replay_memory[:][1])

    output = Variable(torch.Tensor(output),requires_grad=True)
    target = Variable(torch.Tensor(target))
    optimizer.zero_grad()
    loss = net.crit(output,target)
    loss.backward()
    optimizer.step()
    # if i%100==0:
    #     reward.append(t_reward/100)
    #     num_steps.append(sum_j/100)
    #     t_reward = 0
    #     sum_j = 0

evaluate(env,net)