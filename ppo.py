import torch
import torch.nn.functional as F
import numpy as np 
from tensorboardX import SummaryWriter
import torch.nn as nn
import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv 
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from model_and_utils import *
from itertools import count


class PPO():
	def __init__(self, lr, agents, env, eval_env, init_ob):
		self.agents = agents
		self.env = env
		self.eval_env = eval_env
		self.network = actorCritic()
		self.old_network = actorCritic()
		self.dic_placeholder = self.network.state_dict()
		self.old_network.load_state_dict(self.dic_placeholder)
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
		self.last_ob = init_ob
	def experience(self, steps):
		total_obs = np.zeros((steps, self.agents, self.last_ob.shape[1]))
		total_rewards = np.zeros((steps, self.agents))
		total_actions = np.zeros((steps, self.agents))
		total_values = np.zeros((steps+1, self.agents))
		masks = np.zeros((steps, self.agents))

		for step in range(steps):
			total_obs[step] = self.last_ob
			experience_probs, values = self.network(torch.from_numpy(self.last_ob).type(torch.FloatTensor))
			total_values[step] = values.view(-1).detach().numpy()
			m = Categorical(experience_probs)
			actions = m.sample()
			self.last_ob, rews, dones_, _ = self.env.step(actions.numpy())
			dones = np.logical_not(dones_)*1
			total_rewards[step] = rews
			total_actions[step] = actions.numpy()
			masks[step] = dones
		_, values = self.network(torch.from_numpy(self.last_ob).type(torch.FloatTensor))
		total_values[steps] = values.view(-1).detach().numpy()

		advantage, real_values = gae(total_rewards, masks, total_values)
		advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

		return(total_obs, total_values, total_rewards, total_actions, masks, advantage, real_values)
	def eval(self):
		ob = self.eval_env.reset()
		#self.eval_env.render('rgb_array')
		eval_rewards = 0
		for e in count():
			eval_probs, _ = self.network(torch.from_numpy(ob).type(torch.FloatTensor))
			eval_m = Categorical(eval_probs)
			eval_action = eval_m.sample()
			ob, eval_rews, done, _ = self.eval_env.step(eval_action.numpy())
			#self.eval_env.render('rgb_array')

			eval_rewards += eval_rews
			if done:
				break
		return(eval_rewards)		



	def update(self, epochs, steps, total_obs, total_actions, advantage, real_values):


		total_obs_ = torch.from_numpy(total_obs).view(steps, -1, 1).type(torch.FloatTensor)
		total_actions = total_actions.reshape(steps, -1)
		advantage_ = torch.from_numpy(advantage).view(steps, -1).type(torch.FloatTensor)
		real_values_ = torch.from_numpy(real_values).view(steps, -1).type(torch.FloatTensor)



		for _ in range(epochs):
			inds = np.arange(steps)
			np.random.shuffle(inds)

			for t in range(steps):
				index = inds[t]

				probs, values_to_backprop = self.network(total_obs_[index].view(-1))

				m = Categorical(probs)
				action_taken_prob = probs[total_actions[index]]
				entropy = m.entropy()

				old_probs, _ = self.old_network(total_obs_[index].view(-1))
				old_action_taken_probs = old_probs[total_actions[index]]
				old_probs.detach()
				ratios = action_taken_prob/(old_action_taken_probs + 1e-5)

				surr1 = ratios * advantage_[index]
				surr2 = torch.clamp(ratios, min=(1.-.1), max=(1.+.1))*advantage_[index]
				policy_loss = -torch.min(surr1, surr2)
				value_loss = ((values_to_backprop-real_values_[index])**2)
				#value_loss = F.smooth_l1_loss(values_to_backprop, real_values_[index])
				total_loss = policy_loss+value_loss-0.01*entropy

				self.optimizer.zero_grad()
				total_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
				self.optimizer.step()


		self.old_network.load_state_dict(self.dic_placeholder)
		self.dic_placeholder = self.network.state_dict()
		return (value_loss)