import random
import numpy as np
from getDataset import *
# import argparse
from pathlib import Path

# parser = argparse.ArgumentParser()
# parser.add_argument("n", type=np.int32, help="Number of vehicles")
# args = parser.parse_args()

def normalized(n,n_max=6,transitionModel='Markovian'):
	# Create environment
	n = n
	n_max = n_max
	dict1 = get_Dataset(n,transitionModel)

	# For computing reward
	sampleSize = 500

	cw1List = [16,32,48,64,96,128,256,512] # [16,32,48,64,96,128,256]
	cw2List = [32,64,128,256,512] # [32,128]
	actionDim = len(cw1List)

	new_data = []

	for i in range(actionDim):
		for j in range(len(cw2List)):
			key = str(cw1List[i])+'+'+str(cw2List[j])
			data = np.asarray(dict1[key])

			for k in range(sampleSize):
				new_data.append(data[k,:-1])

	new_data = np.asarray(new_data)

	data_min = new_data.min(axis=0)   # 计算每一列的最小值。
	data_max = new_data.max(axis=0)   # 计算每一列的最大值。

	# data_mean = np.mean(new_data,0) # 计算每一列的均值.
	# data_std = np.std(new_data,0)   # std有偏估计计算结果.

	# simulation 1:
	#baseFolder = './Dataset/dataStats/simulation1/vel_'+str(n)+'/'
	# simulation 2:
	baseFolder = './Dataset/dataStats/simulation2/maxVehicle_'+str(n_max)+'/'+'/vel_'+str(n)+'/'

	Path(baseFolder).mkdir(parents=True, exist_ok=True)

	np.savetxt(baseFolder+'data_min.txt',data_min,delimiter =  ',')
	np.savetxt(baseFolder+'data_max.txt',data_max,delimiter =  ',')

# print("begin")
# for n_max in range(2,11):
# 	for n in range(2,n_max+1):
# 		normalized(n,n_max,'NonMarkovian') # n_max from 2 to 10.
# print("end")