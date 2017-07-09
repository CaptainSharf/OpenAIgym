import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.optim as optim
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.l1 = nn.Linear(16,4)
        self.crit = nn.MSELoss()
    def forward(self, x):
        x = self.l1(x)
        return x


replay_memory=[]
gamma = 0.99
#Instantiated Neural Net
net = model()


optimizer = optim.SGD(net.parameters(),lr=0.1)
num_steps = []
reward = []
num_episodes = 2000
env = gym.make('FrozenLake-v0')
epsilon = 1
for i in range(1,num_episodes+1):
    s = env.reset()
    d = False
    j=0
    t_reward = 0
    epsilon = 0.1
    while j<99:
        inp = np.identity(16)[s:s+1]
        inp = Variable(torch.Tensor(inp))
        out  = net.forward(inp)
        Q_val,act = torch.max(out,1)
        act  = act.data.numpy()
        act = act[0][0]

        if np.random.random(1) < epsilon:
            act = env.action_space.sample()
        s1,r,d,_ = env.step(act)

        t_reward+=r
        inp1 = np.identity(16)[s1:s1+1]
        inp1 = Variable(torch.Tensor(inp1))
        out  = net.forward(inp1)
        Q_new,_ = torch.max(out,1)
        val  = r+ gamma*Q_new
        target = Variable(torch.Tensor(val.data),requires_grad=False)
        optimizer.zero_grad()
        loss = net.crit(Q_val,target)
        loss.backward()
        optimizer.step()
        s=s1
        j+=1
        if d==True:
            break
            epsilon = 1./((i/50) + 10)
    num_steps.append(j)
    reward.append(t_reward)

print sum(reward)/num_episodes
plt.plot(reward)
plt.show()
plt.plot(num_steps)
plt.show()