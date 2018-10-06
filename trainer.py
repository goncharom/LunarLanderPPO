import torch
import torch.functional as F
import numpy as np 
import tensorboardX
import torch.nn as nn
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
import matplotlib.pyplot as plt
from model_and_utils import *
from torch.distributions import Categorical


#Parameters
iterations = 50 
steps = 20
epochs = 3
num_processes = 4






env = SubprocVecEnv([make_env(i, 'LunarLander-v2') for i in range(num_processes)]) 
obs_ = env.reset()
network = actorCritic()
old_network = actorCritic()
dic_placeholder = network.state_dict()
old_network.load_state_dict(dic_placeholder)
total_obs = torch.zeros((steps,num_processes, obs_.shape[1]))#Yo que se
total_rewards = torch.zeros((steps, num_processes))
total_actions = torch.zeros((steps, num_processes))
total_values = torch.zeros((steps+1, num_processes))
masks = torch.zeros((steps, num_processes))
last_ob = np.zeros((1, num_processes, obs_.shape[1]))
optimizer = torch.optim.Adam(network.parameters(), lr=2.5e-4)


rewards_to_plot = np.zeros((1, iterations*steps))


_, values = network(torch.from_numpy(obs_))

for _ in range(iterations): #Dont know if this is the most apropiate way
	for step in range(steps): 
	


		total_obs[step].copy_(torch.from_numpy(obs_).type(torch.FloatTensor))
		total_values[step].copy_(values.view(-1))


		actions_probs, _ = network(torch.from_numpy(obs_))
		m = Categorical(actions_probs)
		actions = m.sample()

		obs_, rews, dones_, _ = env.step(actions.numpy())
		dones = np.logical_not(dones_)*1
		_, values = network(torch.from_numpy(obs_))

		total_rewards[step].copy_(torch.from_numpy(rews).type(torch.FloatTensor)) 
		total_actions[step].copy_(actions)
		masks[step].copy_(torch.from_numpy(dones).type(torch.FloatTensor))
		rewards_to_plot[0][step] = total_rewards[step].mean()



	total_obs[steps-1].copy_(torch.from_numpy(obs_).type(torch.FloatTensor))
	total_values[steps-1].copy_(values.view(-1))
	last_ob = obs_


	advantage, real_values = gae(total_rewards, masks, total_values)
	advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

	for _ in range(epochs):
		
		inds = np.arange(steps)
		np.random.shuffle(inds)

		for t in range(steps): #MAL, ES PROB DE ACCION TOMADA
			index = inds[t]
			probs, _ = network(total_obs[index])
			m = Categorical(probs)

			entropy = m.entropy()

			old_probs, _ = old_network(total_obs[index])
			m_old = Categorical(old_probs)
			old_probs.detach()

			ratios = probs /  (old_probs + 1e-5)

			surr1 = ratios * advantage[index]
			surr2 = torch.clamp(ratios, min = (1+.1), max = (1+.1)) * advantage[index]

			policy_loss = -torch.min(surr1, surr2).mean()
			value_loss = (.5*(total_values[index] - real_values[index])).mean()

			total_loss = policy_loss + value_loss - .01 * entropy.mean() 

			optimizer.zero_grad()
			total_loss.backward()#<-Problem 
			torch.nn.utils.clip_grad_norm_(network.parameters(), .5)
			optimizer.step()


	old_network.load_state_dict(dic_placeholder)
	dic_placeholder = network.state_dict()
	obs_ = last_ob 


plotRewards(iterations, rewards_to_plot)