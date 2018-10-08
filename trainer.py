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
iterations = 100
steps = 200
epochs = 3
num_processes = 4






env = SubprocVecEnv([make_env(i, 'LunarLander-v2') for i in range(num_processes)]) 
obs_ = env.reset()
network = actorCritic()
old_network = actorCritic()
dic_placeholder = network.state_dict()
old_network.load_state_dict(dic_placeholder)
"""
total_obs = torch.zeros((steps,num_processes, obs_.shape[1]))#Yo que se
total_rewards = torch.zeros((steps, num_processes))
total_actions = torch.zeros((steps, num_processes))
total_values = torch.zeros((steps+1, num_processes))
"""
total_obs = np.zeros((steps, num_processes, obs_.shape[1]))
total_rewards = np.zeros((steps, num_processes))
total_actions = np.zeros((steps, num_processes))
total_values = np.zeros((steps+1, num_processes))
masks = np.zeros((steps, num_processes))

last_ob = np.zeros((1, num_processes, obs_.shape[1]))
optimizer = torch.optim.Adam(network.parameters(), lr=2.5e-4)


rewards_to_plot = []

_, values = network(torch.from_numpy(obs_))

for iteration in range(iterations): #Dont know if this is the most apropiate way
	print (iteration)
	for step in range(steps): 
	


		#total_obs[step].copy_(torch.from_numpy(obs_).type(torch.FloatTensor))
		total_obs[step] = obs_
		#total_values[step].copy_(values.view(-1))
		total_values[step] = values.view(-1).detach().numpy()


		actions_probs, _ = network(torch.from_numpy(obs_))
		m = Categorical(actions_probs)
		actions = m.sample()

		obs_, rews, dones_, _ = env.step(actions.numpy())
		dones = np.logical_not(dones_)*1
		_, values = network(torch.from_numpy(obs_))

		
		#total_rewards[step].copy_(torch.from_numpy(rews).type(torch.FloatTensor)) 
		total_rewards[step] = rews
		#total_actions[step].copy_(actions)
		total_actions[step] = actions.numpy()
		masks[step] = dones

		rewards_to_plot.append(total_rewards[step].mean())

	plotRewards(rewards_to_plot)


	#total_obs[steps-1].copy_(torch.from_numpy(obs_).type(torch.FloatTensor))
	total_obs[steps-1] = obs_
	#total_values[steps-1].copy_(values.view(-1))
	total_values[steps-1] = values.view(-1).detach().numpy()
	last_ob = obs_


	advantage, real_values = gae(total_rewards, masks, total_values)
	advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

	for _ in range(epochs):
		
		inds = np.arange(steps)
		np.random.shuffle(inds)

		for t in range(steps): #MAL, ES PROB DE ACCION TOMADA
			index = inds[t]
			probs, values = network(torch.from_numpy(total_obs[index]).type(torch.FloatTensor))
			m = Categorical(probs)
			action_probs = torch.zeros((1, num_processes))
			for agent in range(action_probs.size()[1]):

				action_probs[0, agent] = probs[agent, int(total_actions[index, agent])]


			entropy = m.entropy()

			old_probs, _ = old_network(torch.from_numpy(total_obs[index]).type(torch.FloatTensor))
			m_old = Categorical(old_probs)
			old_action_probs = torch.zeros((1, num_processes))
			for agent in range(old_action_probs.size()[1]):
				old_action_probs[0, agent] = old_probs[agent, int(total_actions[index, agent])]


			ratios = action_probs /  (old_action_probs + 1e-5)

			surr1 = ratios * torch.from_numpy(advantage[index]).type(torch.FloatTensor)
			surr2 = torch.clamp(ratios, min = (1+.1), max = (1+.1)) * torch.from_numpy(advantage[index]).type(torch.FloatTensor)

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


