import torch
import gym
import numpy as np
from model_and_utils import *
from torch.distributions import Categorical
from itertools import count


network = actorCritic()
network.load_state_dict(torch.load('./modelo'))
env = gym.make('LunarLander-v2')
obs_= env.reset()
env.render(mode='rgb_array')
for t in count():
	obs = torch.from_numpy(obs_).type(torch.FloatTensor)
	probs, _ = network(obs)
	print (probs)
	m = Categorical(probs)
	action = m.sample()
	obs_, _, done, _ = env.step(action.numpy())
	env.render(mode='rgb_array')
	if done:
		obs_ = env.reset()


