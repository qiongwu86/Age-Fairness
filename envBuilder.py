from simulation import *

def envBuilder(n,ps=1.0,transitionModel='Markovian',history=0,epLen=200,arrival_rate=3,leave_rate=3,n_max=6):
	epLen = epLen            # episode的步长数。
	a_r = arrival_rate       # 车辆到达率。
	l_r = leave_rate         # 车辆离开率。
	actionDict = dict()      # dict()函数创建空字典。
	otherActionDict = dict() # dict()函数创建空字典。

	if transitionModel == "Markovian":
		actions = [16,32,48,64,96,128,256]          # agent车辆的动作空间。[16,32,48,64,96,128,256]
		# num_action = len(actions)
		for i in range(len(actions)):
			actionDict[i] = actions[i]

		otherActions = [32,128,0]                   # 网络其他车辆的动作空间。
		for i in range(len(otherActions)):
			otherActionDict[i] = otherActions[i]

	elif transitionModel == "NonMarkovian":
		actions = [16,32,48,64,96,128,256,512]      # agent车辆的动作空间。
		# num_action = len(actions)
		for i in range(len(actions)):
			actionDict[i] = actions[i]

		otherActions = [32,64,128,256,512,0]        # 网络其他车辆的动作空间。
		for i in range(len(otherActions)):
			otherActionDict[i] = otherActions[i]

	print("Window dictionary initialize completed.")

	env = commEnv(n,ps,transitionModel,actionDict,otherActionDict,epLen,a_r,l_r,n_max,history)

	return env