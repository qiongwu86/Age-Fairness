import numpy as np 
import random
import pickle
import pdb
from math import *
import seaborn as sns
import pandas as pd
from dealData import *
import matplotlib.pyplot as plt
from pandas.plotting import table
from pathlib import Path
from matplotlib.font_manager import FontProperties  
# plt.rc("font",family="SimHei",size="15")  #解决中文乱码问题


def plot_data(file_type,data_path,fig_path):

	if file_type == 'all_rewards':
		all_rewards = load_data(data_path)
		x_len = len(all_rewards)
		x_range = [x for x in range(1,x_len+1)]

		plt.plot(x_range, all_rewards, label='Extensions Traning All Rewards')
		plt.plot(x_range, [1]*x_len, label='Limit')
		plt.grid(linestyle=':')
		plt.legend()
		plt.xlabel('Episodes')
		plt.ylabel('All rewards')
		plt.show()
		# fig.savefig(fig_path, format='eps', dpi=1000)

	elif file_type == 'losses':
		losses = load_data(data_path)
		x_len = len(losses)
		x_range = [x for x in range(1,x_len+1)]

		plt.plot(x_range, losses, label='Extensions DQN Loss')
		plt.grid(linestyle=':')
		plt.legend()
		plt.xlabel('All steps')
		plt.ylabel('Losses')
		plt.show()
		# fig.savefig(fig_path, format='eps', dpi=1000)

def box_plot(data):
	num = 1000
	opt100 = list(data[0])
	opt75 = list(data[1])
	rl100 = list(data[2])
	rl75 = list(data[3])
	rf100 = list(data[4])
	rf75 = list(data[5])
	dt100 = list(data[6])
	dt75 = list(data[7])
	std16_100 = list(data[8])
	std16_75 = list(data[9])
	std32_100 = list(data[10])
	std32_75 = list(data[11])
	std64_100 = list(data[12])
	std64_75 = list(data[13])
	std128_100 = list(data[14])
	std128_75 = list(data[15])
	std256_100 = list(data[16])
	std256_75 = list(data[17])
	std512_100 = list(data[18])
	std512_75 = list(data[19])

	approch_opt = ['OPT' for _ in range(num)]
	approch_rl = ['RL' for _ in range(num)]
	approch_rf = ['RF' for _ in range(num)]
	approch_dt = ['DT' for _ in range(num)]
	approch_std16 = ['SP16' for _ in range(num)]
	approch_std32 = ['SP32' for _ in range(num)]
	approch_std64 = ['SP64' for _ in range(num)]
	approch_std128 = ['SP128' for _ in range(num)]
	approch_std256 = ['SP256' for _ in range(num)]
	approch_std512 = ['SP512' for _ in range(num)]
	ps100 = ['ps=1.0' for _ in range(num)]
	ps75 = ['ps=0.75' for _ in range(num)]

	opt_d100 = pd.DataFrame({'reward': opt100, 'approch': approch_opt, 'ps': ps100})
	opt_d75 = pd.DataFrame({'reward': opt75, 'approch': approch_opt, 'ps': ps75})
	rl_d100 = pd.DataFrame({'reward': rl100, 'approch': approch_rl, 'ps': ps100})
	rl_d75 = pd.DataFrame({'reward': rl75, 'approch': approch_rl, 'ps': ps75})
	rf_d100 = pd.DataFrame({'reward': rf100, 'approch': approch_rf, 'ps': ps100})
	rf_d75 = pd.DataFrame({'reward': rf75, 'approch': approch_rf, 'ps': ps75})
	dt_d100 = pd.DataFrame({'reward': dt100, 'approch': approch_dt, 'ps': ps100})
	dt_d75 = pd.DataFrame({'reward': dt75, 'approch': approch_dt, 'ps': ps75})
	std16_d100 = pd.DataFrame({'reward': std16_100, 'approch': approch_std16, 'ps': ps100})
	std16_d75 = pd.DataFrame({'reward': std16_75, 'approch': approch_std16, 'ps': ps75})
	std32_d100 = pd.DataFrame({'reward': std32_100, 'approch': approch_std32, 'ps': ps100})
	std32_d75 = pd.DataFrame({'reward': std32_75, 'approch': approch_std32, 'ps': ps75})
	std64_d100 = pd.DataFrame({'reward': std64_100, 'approch': approch_std64, 'ps': ps100})
	std64_d75 = pd.DataFrame({'reward': std64_75, 'approch': approch_std64, 'ps': ps75})
	std128_d100 = pd.DataFrame({'reward': std128_100, 'approch': approch_std128, 'ps': ps100})
	std128_d75 = pd.DataFrame({'reward': std128_75, 'approch': approch_std128, 'ps': ps75})
	std256_d100 = pd.DataFrame({'reward': std256_100, 'approch': approch_std256, 'ps': ps100})
	std256_d75 = pd.DataFrame({'reward': std256_75, 'approch': approch_std256, 'ps': ps75})
	std512_d100 = pd.DataFrame({'reward': std512_100, 'approch': approch_std512, 'ps': ps100})
	std512_d75 = pd.DataFrame({'reward': std512_75, 'approch': approch_std512, 'ps': ps75})

	data = pd.concat([opt_d100, opt_d75, rl_d100, rl_d75, rf_d100, rf_d75, dt_d100, dt_d75, std64_d100, std64_d75, std128_d100, std128_d75, std256_d100, std256_d75, std512_d100, std512_d75]) # std32_d100, std32_d75, , std256_d100, std256_d75 
	fig, ax = plt.subplots()
	ax = sns.boxplot(x="approch", y="reward", data=data, hue="ps", showfliers=False, showmeans=False, width=0.4, linewidth=1.0, palette=['red','dodgerblue'])
	for i in [0,2,4,6,8,10,12,14]:	
		mybox = ax.artists[i]
		#mybox.set_facecolor('red')
		mybox.set_edgecolor('red')
		mybox.set_linewidth(1.0)
	for i in [1,3,5,7,9,11,13,15]:	
		mybox = ax.artists[i]
		#mybox.set_facecolor('dodgerblue')
		mybox.set_edgecolor('dodgerblue')
		mybox.set_linewidth(1.0)
	ax.set_xlabel('Approach')
	ax.set_ylabel('Age fairness utility')
	ax.set_ylim([0.7,1])
	# ax.set_title('Comparison of approach fairness'+'\n'+'(Max device = 6, arrival rate = leave rate = 3)')
	ax.grid(linestyle=':')
	plt.show()

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float, axis=0)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def get_all_data(episode_num,step_num,ps,file_type,txt_num,std_init=''):
	all_data = []
	if std_init == '':
		for txt_number in range(1,txt_num+1):
			file_name = str(episode_num)+'_'+str(step_num)+'_'+str(int(ps*100))
			data_path = './Dataset/plotDatas/'+file_type+'/'+file_name+'/'+file_name+'_'+str(txt_number)+'.txt'
			temp_data = load_data(data_path)
			all_data.append(temp_data)
		return np.array(all_data)
	else:
		for txt_number in range(1,txt_num+1):
			file_name = str(episode_num)+'_'+str(step_num)+'_'+str(int(ps*100))
			data_path = './Dataset/plotDatas/'+file_type+'/'+file_name+'/'+str(std_init)+'/'+file_name+'_'+str(txt_number)+'.txt'
			temp_data = load_data(data_path)
			all_data.append(temp_data)
		return np.array(all_data)


if __name__ == '__main__':
	transitionModel = 'NonMarkovian'
	history = 0

	# # traning plot
	episode_num_1,step_num_1,ps_1,file_type_r,txt_num1 = 1000,200,1.0,transitionModel+'/'+'all_rewards',10 # 'all_rewards''test_data'
	episode_num_2,step_num_2,ps_2,file_type_r,txt_num2 = 1000,200,0.75,transitionModel+'/'+'all_rewards',10
	all_rewards_1 = np.sum(get_all_data(episode_num_1,step_num_1,ps_1,file_type_r,txt_num1),axis=0)
	all_rewards_2 = np.sum(get_all_data(episode_num_2,step_num_2,ps_2,file_type_r,txt_num2),axis=0)
	avg_rewards_1 = all_rewards_1/txt_num1
	avg_rewards_2 = all_rewards_2/txt_num2

	x_len = len(avg_rewards_2)
	x_range = [x for x in range(1,x_len+1)]

	font = FontProperties(fname="C:/Windows/Fonts/SimSun.ttc", size=22) 
	font1 = {'family' : 'SimSun',
	'weight' : 'normal',
	'size'   : 22,}

	figure, ax = plt.subplots(figsize=(7.5, 6))

	#设置坐标刻度值的大小以及刻度值的字体
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Times New Roman') for label in labels]

	plt.plot(x_range, [1]*x_len, color='red', label='Absolute fair limit')
	plt.plot(x_range, avg_rewards_1, label='ps=1.0')
	plt.plot(x_range, avg_rewards_2, label='ps=0.75')
	plt.grid(linestyle=':')
	plt.legend()
	plt.xlabel('Episodes')
	plt.ylabel('Age fairness utility')
	plt.title('Extensions DQN training rewards')
	plt.show()


	# box plot
	psList = [1.0,0.75] # 1.0,0.75
	txtList = ['OPT','RL','RF','DT','SP16','SP32','SP64','SP128','SP256','SP512'] #'OPT','RL','RF','DT','SP16','SP32','SP64','SP128','SP256','SP512'

	#psList = [0.75]
	#txtList = ['RL']

	plotData = []

	for file in txtList:
		for ps in psList:
			filePath = './test_results/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/"
			dataPath = filePath + file + '.txt'
			#deal_data(dataPath)
			data = np.loadtxt(dataPath)
			plotData.append(data)

	box_plot(plotData)

