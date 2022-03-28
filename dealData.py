import numpy as np 
import random
import pickle
import pdb
from math import *
import matplotlib.pyplot as plt 
from pathlib import Path


def deal_data(data_path):

	fi = open(data_path,'r+')
	data = fi.read()
	data1 = eval(data)
	data1_len = len(data1)

	for i in range(data1_len):
		if i != (data1_len-1):
			fi.write(str(data1[i]) + '\n')
		else:
			fi. write(str(data1[i]))

	fi.close()

def load_data(data_path):
	all_data = np.loadtxt(data_path)
	# print(all_data,type(all_data),np.shape(all_data))
	return all_data

