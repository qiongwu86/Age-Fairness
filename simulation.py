import numpy as np
import os
import sys
import random
from math import *
from getDataset import *

class commEnv:
	def __init__(self,n,ps,transitionModel,action_dict,other_action_dict,epLen = 200,arrival_rate = 3,leave_rate = 3,n_max=6,history=0):
		"""
		args: T     - Simulation time in seconds
			  n     - Number of vehicles in the network
			  cwMin - Minimum contention window duration
			  cwMax - maximum contention window duration
			  ifs   - Interframe space
			  L     - maximum length of message
		"""
		self.n_max = n_max                                 # 网络中车辆上限.
		self.n_min = 1                                     # 网络中车辆下限.
		self.n = random.randint(self.n_min,self.n_max)     # 网络中车辆数目。
		self.transitionModel = transitionModel             # other vehicle的状态转移模型.
		self.ps = ps                                       # other vehicle的状态转移概率.
		self.countStepsMax = epLen                         # episode的步长.
		self.arrival_rate = arrival_rate                   # 车辆到达率.
		self.leave_rate = leave_rate                       # 车辆离开率.
		self.actionDict = action_dict                      # vehicle1的动作空间.
		self.otherActionDictTotal = other_action_dict      # other vehicles的所有动作空间.
		self.otherActionDict = {i:self.otherActionDictTotal[i] for i in list(self.otherActionDictTotal.keys())[:-1]}  # other vehicles的动作空间.

		self.dimState = 4                                                           # State: ak,ok,W0k,n
		self.dict = get_Dataset(self.n,self.transitionModel)                        # 获取年龄训练数据字典.
		
		if self.transitionModel == 'Markovian':
			baseFolder = './Dataset/dataStats/simulation1/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.
		elif self.transitionModel == 'NonMarkovian':
			baseFolder = './Dataset/dataStats/simulation2/'+'maxVehicle_'+str(self.n_max)+'/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.

		# Count for no. of steps in one episode
		self.countSteps = 0                                # 记录每条 episode 运行的步数.
		self.otherActionIndex = random.choice(list(self.otherActionDict.keys()))
		self.otherActionIndexTemp, self.otherActionIndexBackup = self.otherActionIndex, self.otherActionIndex
		self.changeFlag1 = False
		self.changeFlag2 = False
		self.incrementFlag = random.choice([True, False])

		self.historyFlag = history

		if self.historyFlag:
			self.historyData = []
			self.historyTestData = []

	def preProcess(self,data):
		# Data normalization
		# data1 = np.divide(data-self.data_mean,self.data_std)
		if self.n == 1:
			data1 = []
			for i in range(self.dimState):
				if (self.data_max[i] == 0) and (self.data_min[i] == 0):
					data1.append(0)
				else:
					data1.append(np.divide(data[i]-self.data_min[i],(self.data_max[i]-self.data_min[i])))
			data1 = np.asarray(data1)
		else:
			data1 = np.divide(data-self.data_min,(self.data_max-self.data_min))
		return data1

	def computeReward(self,rhoOmegaDiff):
		reward = (1-rhoOmegaDiff)
		return reward

	def reset(self):
		# 重置网络中车辆数目:
		arriving_vehicle = np.random.poisson(lam=self.arrival_rate)
		leaving_vehicle = np.random.poisson(lam=self.leave_rate)
		self.n = self.n + arriving_vehicle - leaving_vehicle
		self.changeFlag1 = False
		self.changeFlag2 = False

		# 网络中车辆数目判断:
		if self.n > self.n_max:
			self.n = self.n_max
		elif self.n < self.n_min:
			self.n = self.n_min

		# 重置 Episode Step:
		self.countSteps = 0

		# 重置数据字典:
		self.dict = get_Dataset(self.n,self.transitionModel)  # 获取年龄训练数据字典.

		if self.transitionModel == 'Markovian':
			baseFolder = './Dataset/dataStats/simulation1/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.
		elif self.transitionModel == 'NonMarkovian':
			baseFolder = './Dataset/dataStats/simulation2/'+'maxVehicle_'+str(self.n_max)+'/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.

		if self.n == 1:
			#stRaw = self.dict[str(16)+'+'+str(0)]
			stRaw = self.dict[str(random.choice(self.actionDict))+'+'+str(0)]
			self.otherActionIndex = list(self.otherActionDictTotal.keys())[-1]
			stNormalized = self.preProcess(np.asarray(stRaw[0:self.dimState]))
			# Get State
			st = stNormalized
			return st
		else:
			self.otherActionIndex = random.choice(list(self.otherActionDict.keys()))
			self.otherActionIndexTemp, self.otherActionIndexBackup = self.otherActionIndex, self.otherActionIndex
			stRaw = random.choice(self.dict[str(random.choice(self.actionDict))+'+'+str(self.otherActionDict[self.otherActionIndex])])
			stNormalized = self.preProcess(np.asarray(stRaw[0:self.dimState])) # 获取初始化状态的前四个数值。[0,1,2,3,4] 0->3.
			# Get state
			st = stNormalized
			return st

	def step(self,a):
		self.countSteps += 1
		# print('改变前',self.n,self.otherActionIndex,self.otherActionIndexTemp,self.otherActionIndexBackup)
		self.change_env_state()
		# print('改变后',self.n,self.otherActionIndex,self.otherActionIndexTemp,self.otherActionIndexBackup)

		if self.n == 1:
			# print('Key = ',self.otherActionIndex)
			key = str(self.actionDict[a])+'+'+str(0)
			#key = str(16)+'+'+str(0)
			#key = str(random.choice(self.actionDict))+'+'+str(0)

			next_state_full = self.dict[key]
			next_state_normalized = self.preProcess(np.asarray(next_state_full[0:self.dimState]))

			done = False
			reward1 = 1
			info = 0

			# 重置 episode step 数.
			if (self.countSteps >= self.countStepsMax):
				done = True
				self.countSteps = 0

			if self.historyFlag:
				next_state = np.concatenate([next_state_normalized,self.historyData],axis=0)

				# Append current observation to history.
				self.historyData = np.roll(self.historyData,self.dimState)
				self.historyData[0:self.dimState] = np.copy(next_state_normalized)
			else:
				next_state = next_state_normalized

			return (next_state,reward1,done,info)
		else:
			# print('Key = ',self.otherActionIndex)
			# try:
			key = str(self.actionDict[a])+'+'+str(self.otherActionDict[self.otherActionIndex])
			# except KeyError:
			# 	print(self.actionDict)
			# 	print(self.otherActionDict)
			# 	print(self.otherActionIndex)
			# 	print(self.otherActionDict[self.otherActionIndex])

			next_state_full = random.choice(self.dict[key])
			next_state_normalized = self.preProcess(np.asarray(next_state_full[0:self.dimState]))

			done = False
			reward1 = self.computeReward(next_state_full[-1])
			info = 0

			# 重置 episode step 数.
			if (self.countSteps >= self.countStepsMax):
				done = True
				self.countSteps = 0

			if self.historyFlag:
				next_state = np.concatenate([next_state_normalized,self.historyData],axis=0)

				# Append current observation to history.
				self.historyData = np.roll(self.historyData,self.dimState)
				self.historyData[0:self.dimState] = np.copy(next_state_normalized)
			else:
				next_state = next_state_normalized

			return (next_state,reward1,done,info)

	# Use for rf test
	def step_rf(self,a):
		self.countSteps += 1
		self.change_env_state()

		if self.n == 1:
			key = str(self.actionDict[a])+'+'+str(0)

			next_state_full = self.dict[key]
			next_state_normalized = self.preProcess(np.asarray(next_state_full[0:self.dimState]))

			done = False
			reward1 = 1
			info = 0

			# 重置 episode step 数.
			if (self.countSteps >= self.countStepsMax):
				done = True
				self.countSteps = 0

			if self.historyFlag:
				next_state = np.concatenate([next_state_normalized,self.historyData],axis=0)

				# Append current observation to history.
				self.historyData = np.roll(self.historyData,self.dimState)
				self.historyData[0:self.dimState] = np.copy(next_state_normalized)
			else:
				next_state = next_state_normalized

			return (next_state,reward1,done,info)
		else:
			key = str(self.otherActionDict[a])+'+'+str(self.otherActionDict[self.otherActionIndex])

			next_state_full = random.choice(self.dict[key])
			next_state_normalized = self.preProcess(np.asarray(next_state_full[0:self.dimState]))

			done = False
			reward1 = self.computeReward(next_state_full[-1])
			info = 0

			# 重置 episode step 数.
			if (self.countSteps >= self.countStepsMax):
				done = True
				self.countSteps = 0

			if self.historyFlag:
				next_state = np.concatenate([next_state_normalized,self.historyData],axis=0)

				# Append current observation to history.
				self.historyData = np.roll(self.historyData,self.dimState)
				self.historyData[0:self.dimState] = np.copy(next_state_normalized)
			else:
				next_state = next_state_normalized

			return (next_state,reward1,done,info)

	def change_env_state(self):
		# 网络中总车辆数目变化:
		arriving_vehicle = np.random.poisson(lam=self.arrival_rate)
		leaving_vehicle = np.random.poisson(lam=self.leave_rate)
		n_temp = self.n
		self.n = self.n + arriving_vehicle - leaving_vehicle
		self.changeFlag1 = False
		self.changeFlag2 = False

		# 网络中车辆数目判断:
		if self.n > self.n_max:
			self.n = self.n_max
		elif self.n < self.n_min:
			self.n = self.n_min

		# the number of vehicle in the network change from n to 1.
		if (n_temp != 1) and (self.n == 1):
			self.changeFlag1 = True

		# the number of vehicle in the network change from 1 to n.
		if (n_temp == 1) and (self.n != 1):
			self.changeFlag2 = True

		self.dict = get_Dataset(self.n,self.transitionModel)

		if self.transitionModel == 'Markovian':
			baseFolder = './Dataset/dataStats/simulation1/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.
		elif self.transitionModel == 'NonMarkovian':
			baseFolder = './Dataset/dataStats/simulation2/'+'maxVehicle_'+str(self.n_max)+'/'+'vel_'+str(self.n)+'/'
			self.data_min = np.loadtxt(baseFolder+'data_min.txt',delimiter = ',')   # 获取ak、ok和W0k的最大值[(32+48+64+96+...+512)/9].
			self.data_max = np.loadtxt(baseFolder+'data_max.txt',delimiter = ',')   # 获取ak、ok和W0k的最小值.

		if self.n == 1: # 网络中只剩agent车辆。
			# from 1 to 1.
			self.otherActionIndex = list(self.otherActionDictTotal.keys())[-1]      # 其他车辆窗口CW置0.
		else:           # 网络中车辆数目不为1。
			if self.transitionModel == "Markovian":

				p = np.random.uniform(0,1,1)
				if p < self.ps+0.01:
					self.otherActionIndexTemp = not(self.otherActionIndexTemp)
					self.otherActionIndex = self.otherActionIndexTemp
					self.otherActionIndexBackup = self.otherActionIndexTemp
				elif self.changeFlag2:
					self.otherActionIndex, self.otherActionIndexTemp = self.otherActionIndexBackup, self.otherActionIndexBackup

			elif self.transitionModel == "NonMarkovian":

				p = np.random.uniform(0,1,1)
				# if self.changeFlag1:
				# 	# from n to 1.
				# 	self.otherActionIndexTemp, self.otherActionIndexBackup = self.otherActionIndex, self.otherActionIndex
				if (self.changeFlag1 == False) and (self.changeFlag2 == True):
					# from 1 to n.
					self.otherActionIndex = self.otherActionIndexBackup
					if self.incrementFlag:
						if p < self.ps+0.01:
							if self.otherActionIndex == len(list(self.otherActionDict.keys()))-1:
								self.otherActionIndex -= 1
								self.otherActionIndexBackup = self.otherActionIndex
								self.incrementFlag = False
							else:
								self.otherActionIndex += 1
								self.otherActionIndexBackup = self.otherActionIndex
					else:
						if p < self.ps+0.01:
							if self.otherActionIndex == 0:
								self.otherActionIndex += 1
								self.otherActionIndexBackup = self.otherActionIndex
								self.incrementFlag = True
							else:
								self.otherActionIndex -= 1
								self.otherActionIndexBackup = self.otherActionIndex
				elif (self.changeFlag1 == False) and (self.changeFlag2 == False):
					# from n to n.
					if self.incrementFlag:
						if p < self.ps+0.01:
							if self.otherActionIndex == len(list(self.otherActionDict.keys()))-1:
								self.otherActionIndex -= 1
								self.otherActionIndexBackup = self.otherActionIndex
								self.incrementFlag = False
							else:
								self.otherActionIndex += 1
								self.otherActionIndexBackup = self.otherActionIndex
					else:
						if p < self.ps+0.01:
							if self.otherActionIndex == 0:
								self.otherActionIndex += 1
								self.otherActionIndexBackup = self.otherActionIndex
								self.incrementFlag = True
							else:
								self.otherActionIndex -= 1
								self.otherActionIndexBackup = self.otherActionIndex



