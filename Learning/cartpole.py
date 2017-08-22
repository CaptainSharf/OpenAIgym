import gym
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return F.log_softmax(self.head(x.view(x.size(0), -1)))

resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def getimage(img):
	img = resize(img)
	#img = img.view(1,img.size(0),img.size(1),img.size(2))
	#returns a dimension of size 1 inserted at the specified position,in this case it's 0
	img = img.unsqueeze(0).type(torch.FloatTensor)
	return img

gamma = 0.9
model = DQN()
env = gym.make("CartPole-v0")
num_episodes = 100
optimizer = optim.RMSprop(model.parameters())

for i in range(num_episodes):
	env.reset()
	d  = False
	curr_img = env.render(mode='rgb_array')
	curr_img = getimage(curr_img)
	curr_action_val = model(Variable(curr_img).type(torch.FloatTensor))
	Q_curr,a = torch.max(curr_action_val,1)
	while d==False:
		#print img.size()
		a = a.data.numpy()[0][0]
		print a
		s,r,d,_ = env.step(a)
		if d==False:
			next_img = env.render(mode='rgb_array')
			next_img = getimage(next_img)
			next_action_val = model(Variable(next_img).type(torch.FloatTensor))
			Q_next,a = torch.max(next_action_val,1)

			optimizer.zero_grad()
			nn_sq = Q_curr-(r+gamma*Q_next)
			loss = nn_sq*nn_sq
			loss.backward()
			optimizer.step()

			curr_img = next_img
			curr_action_val = next_action_val
			Q_curr = Q_next