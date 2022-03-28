# -*- coding: utf-8 -*-
import math, random
# import gym
import time
import numpy as np
import argparse
from pathlib import Path

from envBuilder import *
from simulation import *
from getDataset import *
from rainbow import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--otherVehicle_Num", type=np.int32, default=2, help="Number of vehicles")
parser.add_argument("--ps", type=np.float32, default=1.0, help="Transition Probability") # 0.75
parser.add_argument("--transitionModel", type=str, default='Markovian', help="Transition Model") # Markovian
parser.add_argument("--history", type=np.int32, default=0, help="History Level")
args = parser.parse_args()

otherVehicle_Num = args.otherVehicle_Num
ps = args.ps
transitionModel = args.transitionModel
history = args.history

episode_num = 1000
epLen = 200
totalVehicle_Num = otherVehicle_Num + 1
arrival_rate = 3
leave_rate = 3
n_max = 6

env = envBuilder(totalVehicle_Num,ps,transitionModel,history,epLen,arrival_rate,leave_rate,n_max)
num_input = len(env.reset())
print('Input Dimension = ',num_input)
num_action = len(list(env.actionDict.keys()))

baseFolder ='./modelRL_train/'+transitionModel+'/'+str(history)+'history/mymodel/'
Path(baseFolder).mkdir(parents=True, exist_ok=True)
path = baseFolder+str(episode_num)+'_'+str(epLen)+'_'+'ps'+str(int(100*ps))+'_'+time.strftime("%b_%d_%Y_%H_%M_%S",time.localtime(time.time()))+".pt"

dataFolder = './Dataset/plotDatas/'+transitionModel+'/'
Path(dataFolder).mkdir(parents=True, exist_ok=True)
data_path = dataFolder+str(episode_num)+'_'+str(epLen)+'_'+str(int(100*ps))+'_'+time.strftime("%b_%d_%Y_%H_%M_%S",time.localtime(time.time()))

num_atoms = num_action
Vmin = 43
Vmax = 50

current_model = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)
target_model  = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)

if USE_CUDA:
	current_model = current_model.cuda()
	target_model  = target_model.cuda()

optimizer = optim.Adam(current_model.parameters(), 0.0001)

replay_buffer = ReplayBuffer(100000)

update_target(current_model, target_model)

num_frames = episode_num*epLen
batch_size = 32
gamma      = 0.99

losses = []
avg_rewards = []
all_rewards = []
episode_reward = 0

state = env.reset()

for frame_idx in range(1,num_frames + 1):
	action = current_model.act(state)
	next_state, reward, done, _ = env.step(action)
	replay_buffer.push(state, action, reward, next_state, done)
	episode_reward += reward

	if done:
		print('Frame = ',frame_idx,'Episode All Reward = ',episode_reward)
		state = env.reset()
		avg_rewards.append(episode_reward/epLen)
		all_rewards.append(episode_reward)
		episode_reward = 0

	if len(replay_buffer) > batch_size:
		loss = compute_td_loss(batch_size,replay_buffer,Vmax,Vmin,num_atoms,current_model,target_model,optimizer)
		losses.append(loss.data)

	if frame_idx % 200 == 0:
		update_target(current_model, target_model)

torch.save(current_model.state_dict(),path)
np.savetxt(data_path+'.txt', avg_rewards)

