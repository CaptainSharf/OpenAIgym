import numpy as np
import gym
def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    curr = False
    while curr is not True:
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        curr = done
    return totalreward

noise_scaling = 0.1 #random scaling
parameters = np.random.rand(4)*2-1
env = gym.make('CartPole-v0')
final_params = parameters
max_reward = 0
for i in range(10000):
    parameters = final_params+(np.random.rand(4)*2-1)*noise_scaling
    curr_reward = run_episode(env,parameters)
    if max_reward < curr_reward:
        final_params = parameters
        max_reward = curr_reward
    if max_reward==200:
    	break
print max_reward,final_params