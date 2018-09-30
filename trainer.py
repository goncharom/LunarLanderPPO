import torch
import torch.functional as F
import numpy as np 
import tensorboardX
import torch.nn as nn
import gym
from baselinse.common.vec_env.subproc_vec_env import SubprocVecEnv 
from baselines.common.vec_env.vec_frame_stack import VecFrameStack 
import matplotlib.pyplot as plt
from models_and_utils import *
from torch.distributions import Categorical



#Parametros
steps = 500
batch_size = 5
epochs = 3
num_processes = 4
env = SubprocVecEnv([make_env(i, 'LunarLander-v2') for i in range(num_processes)])
env = VecFrameStack(env, num_processes) #num_stack?????

network = actorCritic()
old_network = actorCritic()

dic_placeholder = network.state_dict()
old_network.load_state_dict(dic_placeholder)

for step in range(steps):


