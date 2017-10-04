from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



N = 10
T = 10000
DELTA = 0.1
ETA = np.sqrt((2*np.log(N))/(N*T*np.log(10)))

global weights
weights = np.ones(N)


def bernouli(val):
	if np.random.rand() <val:
		return 1
	return 0

def select_action():
	prob_dist = (1-ETA)*(weights/np.sum(weights))+(ETA/N)
	return [np.random.choice(np.arange(N),None,True,prob_dist),prob_dist]

def get_reward(ep_num):
	reward = [bernouli(0.5) for i in range(8)]
	reward.append(bernouli(0.4))
	reward.append(bernouli(0.6) if ep_num<=T/2 else bernouli(0.3))
	return np.array(reward)

def test_policy(ep_num):
	reward_vec = np.zeros(N)
	act_reward = 0
	for _ in range(200):
		ind,_ = select_action()
		reward_step = get_reward(ep_num)
		act_reward = reward_step[ind]
		reward_vec = reward_vec+reward_step
	return (np.max(reward_vec-act_reward))/200


if __name__ == "__main__":
	regret_list = [[0 for i in range(T)] for j in range(200)]
	time_list = [0]
	for j in range(200):
		act_regret = 0
		np.random.seed(j)
		for ep_num in range(T):
			ind,prob = select_action()
			reward_vec = get_reward(ep_num)
			act_regret = act_regret+(np.max(reward_vec)-reward_vec[ind])
			weights[ind] = weights[ind]*np.exp((ETA*reward_vec[ind])/(N*prob[ind]))
			regret_list[j][ep_num] = (act_regret/(ep_num+1))
		plt.plot(range(T),regret_list[j],'r')

	plt.ylabel('EXP3 average regret')
	plt.xlabel('Time Horizon')
	plt.savefig('EXP3_'+str(T)+'.png')
	plt.show()