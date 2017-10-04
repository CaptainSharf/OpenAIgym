from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

eps = [0.1,0.01,0]
N = 10
T = 1000

emiprical_means = np.array([[0,1] for i in range(N)])
def epsilon_greedy(eps,ep_num):
	if np.random.rand() <eps:
		return np.random.choice(np.arange(N))
	return np.argmax(emiprical_means[:,0]+np.sqrt(2*ep_num/emiprical_means[:,1]))

def get_rewards():
	mu = [6+i for i in range(10)]
	reward_vec = [np.random.normal(mu[i],1) for i in range(10)]
	return reward_vec

def update_mean(ind,reward):
	total_val = (emiprical_means[ind][0])*(emiprical_means[ind][1])+reward
	emiprical_means[ind,1]+=1
	emiprical_means[ind,0] = total_val/emiprical_means[ind,1]

if __name__== "__main__":

	regret_list = [[],[],[]]
	arm_pulled = [[0 for i in range(10)] for j in range(3)]
	for j in range(3):
		reward_vec = np.array([0 for i in range(N)])
		reward_sofar = 0
		for i in range(T):
			step_reward = get_rewards()
			reward_vec = reward_vec+step_reward
			ind = epsilon_greedy(eps[j],i)
			arm_pulled[j][ind]+=1
			reward_sofar = reward_sofar+step_reward[ind]
			update_mean(ind,step_reward[ind])
			regret = (np.max(reward_vec)-reward_sofar)/(i+1)
			regret_list[j].append(regret)

	#Normal Plot
	plt.plot(regret_list[0],'r')
	plt.plot(regret_list[1],'b')
	plt.plot(regret_list[2],'m')
	plt.ylabel('EPS Greedy average regret')
	plt.xlabel('Time Horizon')
	plt.savefig('EPS_'+str(T)+'_'+'.png')

	#Log Plot
	# time_list  = np.log(range(1,T+1)).tolist()
	# plt.plot(time_list,regret_list[0],'r')
	# plt.plot(time_list,regret_list[1],'b')
	# plt.plot(time_list,regret_list[2],'m')
	# plt.ylabel('EPS Greedy average regret')
	# plt.xlabel('Log Time Horizon')
	# plt.savefig('EPS_Log'+str(T)+'_'+'.png')

	#Arm pulled
	# time_list  = np.log(range(1,T+1)).tolist()
	# plt.plot(arm_pulled[0],'r')
	# plt.plot(arm_pulled[1],'b')
	# plt.plot(arm_pulled[2],'m')
	# plt.ylabel('Number of times pulled')
	# plt.xlabel('Arm')
	# plt.savefig('Arm'+str(T)+'_'+'.png')