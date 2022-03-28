import numpy as np 
import random
import pickle
import pdb

def get_Dataset(n,transitionModel):
	n = n
	s_slot = 4
	slot_time = 50e-6

	if transitionModel == 'Markovian':
		cw1List = [16,32,48,64,96,128,256]
		cw2List = [32,128]
	elif transitionModel == 'NonMarkovian':
		cw1List = [16,32,48,64,96,128,256,512]
		cw2List = [32,64,128,256,512]
	actionDim = len(cw1List)

	data_dict = dict()

	dataFolder = 'vel_' + str(n) + '/'

	if n == 1:
		for i in range(actionDim):
			vehicle1RXPackets = ((((cw1List[i]-1)*slot_time)/2)+s_slot*slot_time)
			dataTemp = [vehicle1RXPackets,0,cw1List[i],n,1]

			key = str(cw1List[i])+'+'+str(0)
			data_dict[key] = dataTemp

	else:
		sampleSize = 500
		omega = (1.0/n)*np.ones((sampleSize))

		for i in range(actionDim):
			for j in range(len(cw2List)):

				vehicle1RXPackets = np.loadtxt('./Dataset/'+dataFolder+'vehicle1RXPackets'+'+'+str(cw1List[i])+'+'+str(cw2List[j])+'.txt')

				otherRXPackets = np.loadtxt('./Dataset/'+dataFolder+'otherRXPackets'+'+'+str(cw1List[i])+'+'+str(cw2List[j])+'.txt')

				totalPackets = vehicle1RXPackets + otherRXPackets
				rho = np.divide(vehicle1RXPackets,totalPackets)
				reward = np.abs(rho - omega)

				dataset = []
				for k in range(sampleSize):
					dataTemp = [vehicle1RXPackets[k],otherRXPackets[k],cw1List[i],n,reward[k]]
					dataset.append(dataTemp)

				key = str(cw1List[i])+'+'+str(cw2List[j])
				data_dict[key] = dataset

	return data_dict

