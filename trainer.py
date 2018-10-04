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
optimizer = torch.optim.Adam()





env = SubprocVecEnv([make_env(i, 'LunarLander-v2') for i in range(num_processes)]) #Tambien puedo pasar una lista de envs
obs_ = env.reset()
network = actorCritic()
old_network = actorCritic()
dic_placeholder = network.state_dict()
old_network.load_state_dict(dic_placeholder)
total_obs = torch.zeros((steps,num_processes, obs_.shape[1])) #esto esta fatal, encontrar mejor manera de poner el 8
total_rewards = torch.zeros((steps, num_processes))
total_actions = torch.zeros((steps, num_processes))
total_values = torch.zeros((steps+1, num_processes))
masks = torch.zeros((steps, num_processes))
last_ob = torch.zeros((1, num_processes, obs_.shape[1]))
optimizer = torch.optim.Adam(network.parameters(), lr=2.5e-4)

obs = torch.from_numpy(obs_).type(torch.FloatTensor)
_, values = network(obs)
total_values[0] = values

for _ in range(iterations): #Dont know if this is the most apropiate way
	for step in range(steps): #BUCLE PISA PRIMERA OBS

		actions_probs, _ = network(obs)
		m = Categorical(actions_probs)
		actions = m.sample()

		obs_, rews, dones_, _ = env.step(actions.numpy())
		dones = np.logical_not(dones_)*1
		_, values = network(torch.from_numpy(obs_))

		total_obs[step] = torch.from_numpy(obs_).type(torch.FloatTensor)
		total_rewards[step] = torch.from_numpy(rews).type(torch.FloatTensor)
		total_actions[step] = actions
		masks[step] = torch.from_numpy(dones).type(torch.FloatTensor)
		values[steps+1] = values
	#Save last obs!!!

	last_ob = torch.from_numpy(obs)

	#Update block - Randomize batches, gae, backprop with clipped gradients!!!!!
	#Ver dimensiones matrices en update
	#total_obs = total_obs.view(steps * num_processes, -1)
	advantage, real_values = gae(total_rewards, total_values, masks)
	advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
	#Regularize adv
	inds = range(steps)
	np.random.shuffle(inds)

	for t in range(steps): #No se si se puede hacer asi
		index = inds[t]
		probs, _ = network(total_obs[index])
		m = Categorical(probs)
		entropy = m.entropy()

		old_probs, _ = old_network(total_obs[index])
		m_old = Categorical(old_probs)

		ratios = probs / (old_probs + 1e-5)

		surr1 = ratios * advantage[index]
		surr2 = torch.clamp(ratios, min = (1+.1), max = (1+.1)) * advantage[index]

		policy_loss = -torch.min(surr1, surr2).mean()
		value_loss = (.5*(values[index] - real_values[index])).mean()

		total_loss = policy_loss + value_loss + entropy.mean() #Faltan coef, comprobar signos
		optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm(network.parameters(), .5)
		optimizer.step()


	old_network.load_state_dict(dic_placeholder)
	dic_placeholder = network.state_dict()


