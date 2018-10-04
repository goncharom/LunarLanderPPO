import torch
import torch.nn.functional as F
import numpy as np 
import gym
import torch.nn as nn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
import matplotlib.pyplot as plt

class actorCritic(nn.Module):
	def __init__(self):
		super(actorCritic, self).__init__()

		self.fc1 = nn.Linear(8, 32)
		self.fc2 = nn.Linear(32, 64)
		self.fc3 = nn.Linear(64, 128)

		self.value = nn.Linear(128, 1)
		self.actor = nn.Linear(128, 4)

	def forward(self, inputs):
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		probs = F.softmax(self.actor(x))

		value = F.relu(self.value(x))

		return probs, value


def gae (rewards, masks, values):

	gamma = 0.99
	lambd = 0.95

	T, W = rewards.size()

	advantages = torch.zeros((T, W))

	adv_t = torch.zeros(1, W)
	for t in reversed(range(T)):

		delta = rewards[t] + values[t+1].data * gamma*masks[t] - values[t].data

		adv_t = delta + adv_t*gamma*lambd*masks[t]

		advantages[t] = adv_t

		real_values = values[:T].data + advantages

		return advantages, real_values
def make_env(rank, env_id):
	def env_fn():
		env = gym.make(env_id)
		env.seed(1+rank)
		return env
	return env_fn

def plotRewards(iteration, rewards):
	print (rewards)
	fig = plt.figure()
	plt.plot(iteration, rewards)
	plt.pause(10)




