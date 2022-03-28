import math, random
#import gym
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

from rainbow import *
from envBuilder import *
from getDataset import *
from simulation import *

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--otherVehicle_Num", type=np.int32, default=2, help="Number of vehicles")
parser.add_argument("--ps", type=np.float32, default=1.0, help="Transition Probability")
parser.add_argument("--transitionModel", type=str, default='NonMarkovian', help="Transition Model")
parser.add_argument("--history", type=np.int32, default=0, help="History Level")
args = parser.parse_args()

otherVehicle_Num = args.otherVehicle_Num
ps = args.ps
transitionModel = args.transitionModel
history = args.history

useModelNum = 7
testEpisode = 1000
testEpLen = 200
totalVehicle_Num = otherVehicle_Num + 1
arrival_rate = 3
leave_rate = 3
n_max = 6

sence = 'simulation2'

env = envBuilder(totalVehicle_Num,ps,transitionModel,history,testEpLen,arrival_rate,leave_rate,n_max)
num_input = len(env.reset())
print('Input Dimension = ',num_input,',maxVel: '+str(n_max)+'.')
num_action = len(list(env.actionDict.keys()))

pathRL = './modelRL_train/'+transitionModel+'/'+str(history)+'history/mymodel/'+'ps'+str(int(100*ps))+'/'+'ps'+str(int(100*ps))+'_'+str(useModelNum)+".pt"

basePath = './test_results/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/"
Path(basePath).mkdir(parents=True, exist_ok=True)

num_atoms = num_action
Vmin = 43
Vmax = 50

current_model = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)
target_model  = RainbowDQN(num_input, num_action, num_atoms, Vmin, Vmax)

if USE_CUDA:
    current_model = current_model.cuda()
    target_model  = target_model.cuda()

current_model.load_state_dict(torch.load(pathRL,map_location=torch.device('cpu')))
current_model.eval()

gamma = 0.99

print('pathRL = ',pathRL)

testEpisodeCount = testEpisode
StateListRL = []
OtherList = []
ActionListRL = []
RewardListRL = []
RewardListEP = []

for k in range(testEpisodeCount):

    sRL = env.reset()
    otherState = env.otherActionDictTotal[env.otherActionIndex]
    done = False
    rewardRL = 0
    ind = 0

    stateListRL = []
    otherList = []
    actionListRL = []
    rewardListRL = []

    while not done:
        baseFolder = './Dataset/dataStats/'+sence+'/maxVehicle_'+str(n_max)+'/vel_'+str(env.n)+'/'
        data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')
        data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')

        actRL = current_model.act(sRL)
        s_temp = sRL*(data_max-data_min)+data_min
        stateListRL.append(list(s_temp))
        otherList.append(otherState)

        state,reward,done,_ = env.step(actRL)
        otherState = env.otherActionDictTotal[env.otherActionIndex]
        sRL = state
        actionListRL.append(env.actionDict[actRL])
        rewardListRL.append(reward)
        rewardRL+=reward

    StateListRL.append(stateListRL)
    OtherList.append(otherList)
    ActionListRL.append(actionListRL)
    RewardListRL.append(rewardListRL)
    RewardListEP.append(rewardRL/env.countStepsMax)

    print('Episode:'+str(k+1),'Reward:'+str(rewardRL/env.countStepsMax))

print(StateListRL)
print(OtherList)
print(ActionListRL)
print(RewardListRL)
print(RewardListEP)
print('Average reward (RL) = ',np.mean(RewardListRL))
print('List of rewards stored at: ',basePath)
np.savetxt(basePath+'maxVel'+str(n_max)+".txt",RewardListEP)


