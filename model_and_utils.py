import torch
import torch.nn.functional as F
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn as nn
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
import matplotlib.pyplot as plt
from torch.distributions import Categorical



total_rewards_to_plot = []
total_updates = []
total_means = []
total_value_loss = []
total_value_loss_means = []
class actorCritic(nn.Module):
	def __init__(self):
		super(actorCritic, self).__init__()

		self.fc1 = nn.Linear(8, 16)
		self.fc2 = nn.Linear(16, 16)
		self.critic = nn.Linear(16, 1)
		self.actor = nn.Linear(16, 4)


	def forward(self, inputs):
		x = inputs 
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))

		value = self.critic(x)
		probs = F.softmax(self.actor(x))

		return probs, value


def gae (rewards, masks, values):

	gamma = 0.99
	lambd = 0.95

	T, W = rewards.shape
	real_values = np.zeros((T, W))
	advantages = np.zeros((T, W))

	adv_t = 0
	for t in reversed(range(T)):

		delta = rewards[t]*5e-2 + values[t+1] * gamma*masks[t] - values[t]

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
def plotRewards(rewards, updates_no):
	total_rewards_to_plot.append(rewards)
	total_updates.append(updates_no)
	total_means.append(np.mean(total_rewards_to_plot))
	plt.figure(2)
	plt.clf()
	plt.plot(rewards)
	plt.plot(total_updates, total_rewards_to_plot)
	plt.plot(total_updates, total_means)
	plt.pause(0.001)
def plotValueLoss(valuesLoss):
	total_value_loss.append(float(valuesLoss))
	total_value_loss_means.append(np.mean(total_value_loss))
	plt.figure(1)
	plt.clf()
	plt.plot(total_value_loss_means)
	plt.pause(0.001)