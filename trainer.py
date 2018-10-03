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



#Parametros
steps = 5
batch_size = 4
epochs = 3
num_processes = 4
env = SubprocVecEnv([make_env(i, 'LunarLander-v2') for i in range(num_processes)]) #Tambien puedo pasar una lista de envs
obs_ = env.reset()

network = actorCritic()
old_network = actorCritic()

dic_placeholder = network.state_dict()
old_network.load_state_dict(dic_placeholder)

total_obs = torch.zeros((steps,num_processes, obs_.shape[1])) #esto esta fatal, encontrar mejor manera de poner el 8
total_rewards = torch.zeros((steps, num_processes))
total_actions = torch.zeros((steps, num_processes))
masks = torch.zeros((steps, num_processes))


obs = torch.from_numpy(obs_).type(torch.FloatTensor)

for step in range(steps): #DONT LOSE CONTINUITY, SAVE LAST OBS

	actions_probs, _ = network(obs)
	m = Categorical(actions_probs)
	actions = m.sample()

	obs_, rews, dones_, _ = env.step(actions.numpy())
	dones = np.logical_not(dones_)*1
	total_obs[step] = torch.from_numpy(obs_).type(torch.FloatTensor)
	total_rewards[step] = torch.from_numpy(rews).type(torch.FloatTensor)
	total_actions[step] = actions
	masks[step] = torch.from_numpy(dones).type(torch.FloatTensor)
	#Guardar ultima obs!!!!!!!!
	
	#Bloque update




