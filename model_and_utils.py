import torch
import torch.nn.functional as F
import numpy as np 
import gym
import torch.nn as nn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
import matplotlib.pyplot as plt
#Input dim (224,240,3)
#Definir botontes/combinaciones de botones 
means = []


class actorCritic(nn.Module):
	def __init__(self):
		super(actorCritic, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 8, stride=4)
		self.conv2 = nn.Conv2d(16, 64, 5, stride=2)
		self.conv3 = nn.Conv2d(64, 32, 3)
		self.fc1 = nn.Linear(11*12*32, 512)
		self.value = nn.Linear(512, 1)
		self.actor = nn.Linear(512, 8)
		"""
		self.fc1 = nn.Linear(8, 32)
		self.fc2 = nn.Linear(32, 64)
		self.fc3 = nn.Linear(64, 128)

		self.value = nn.Linear(128, 1)
		self.actor = nn.Linear(128, 4)
		"""
	def forward(self, x):
		x = F.relu(self.conv1(x/255.))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv3(x))
		probs = 
		"""
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		probs = F.softmax(self.actor(x))

		value = F.relu(self.value(x))

		return probs, value
		"""
def buttonsTranslator(actions_vector):
	

def gae (rewards, masks, values):

	gamma = 0.99
	lambd = 0.95

	T, W = rewards.shape

	advantages = np.zeros((T, W))

	adv_t = np.zeros((1, W))
	for t in reversed(range(T)):

		delta = rewards[t] + values[t+1] * gamma*masks[t] - values[t]

		adv_t = delta + adv_t*gamma*lambd*masks[t]

		advantages[t] = adv_t

		real_values = values[:T] + advantages

		return advantages, real_values
def make_env(rank, env_id):
	def env_fn():
		env = gym.make(env_id)
		env.seed(1+rank)
		return env
	return env_fn

def plotRewards(rewards):
	rewards = np.array(rewards)
	means.append(rewards.mean())
	plt.figure(2)
	plt.clf()
	#plt.plot(rewards)
	plt.plot(means)
	plt.pause(0.001)




