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

env = gym.make('CartPole-v0')
final_params = [0 for i in range(4)]
max_reward = 0
for i in range(10000):
    parameters = np.random.rand(4)*2-1
    curr_reward = run_episode(env,parameters)
    if max_reward < curr_reward:
        final_params = parameters
        max_reward = curr_reward
print max_reward,final_params