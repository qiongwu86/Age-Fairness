from simulation import *
import numpy as np
import argparse
import pickle
from getDataset import *
from sklearn.tree import DecisionTreeClassifier     # 决策树
from sklearn.ensemble import RandomForestClassifier # 随机森林
from joblib import dump, load
from envBuilder import *
from pathlib import Path


def getDatasetRF(env,N,num_action):
    state = env.reset()
    X=[]
    y=[]
    R=[]

    for i in range(num_action):
        print('Training i =  ',i+1)
        for j in range(N):
            action = i
            # print('action',action)
            next_state_list = env.step(action)
            X.append(state)
            state = next_state_list[0]
            reward = next_state_list[1]
            y.append(int(env.otherActionIndexTemp)) # env.otherActionIndex
            R.append(reward)
    return X,y,R

# Model parameter input
parser = argparse.ArgumentParser()
parser.add_argument("--otherVehicle_Num", type=np.int32, default=2, help="Number of vehicles")
parser.add_argument("--ps", type=np.float32, default=0.75, help="Transition Probability")
parser.add_argument("--transitionModel", type=str, default='Markovian', help="Transition Model")
parser.add_argument("--history", type=np.int32, default=0, help="History Level")
args = parser.parse_args()

# Create environment
otherVehicle_Num = args.otherVehicle_Num
ps = args.ps
print('ps = ',ps,'100ps = ',int(100*ps))
transitionModel = args.transitionModel
history = args.history
totalVehicle_Num = otherVehicle_Num + 1
arrival_rate = 3
leave_rate = 3
n_max = 6

epLen = 200

env = envBuilder(totalVehicle_Num,ps,transitionModel,history,epLen,arrival_rate,leave_rate,n_max)
num_action = len(list(env.actionDict.keys()))

numTrain = 2000
xTrain,yTrain,rTrain = getDatasetRF(env,numTrain,num_action)

numTest = 500
xTest,yTest,rTest = getDatasetRF(env,numTest,num_action)

baseFolder1 ='./modelDT_train/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/" 
Path(baseFolder1).mkdir(parents=True, exist_ok=True)

baseFolder2 ='./modelRF_train/'+transitionModel+'/'+str(history)+'history/ps'+str(int(100*ps))+"/" 
Path(baseFolder2).mkdir(parents=True, exist_ok=True)

np.savetxt(baseFolder1+'xtrain.txt',xTrain,delimiter = ',')
np.savetxt(baseFolder1+'ytrain.txt',yTrain,delimiter = ',')
np.savetxt(baseFolder1+'rtrain.txt',rTrain,delimiter = ',')

np.savetxt(baseFolder2+'xtrain.txt',xTrain,delimiter = ',')
np.savetxt(baseFolder2+'ytrain.txt',yTrain,delimiter = ',')
np.savetxt(baseFolder2+'rtrain.txt',rTrain,delimiter = ',')


# Training DT
clf = DecisionTreeClassifier(random_state=0) # 20
clf.fit(np.asarray(xTrain), np.asarray(yTrain))
ypred = clf.predict(xTrain)
print('Decision Tree Training Accuracy = ',100*(ypred.shape[0]-np.sum(np.abs(ypred-yTrain)))/ypred.shape[0])

# Training RF
rfc = RandomForestClassifier(random_state=0) #,20 15 n_estimators=20, max_depth=15, 
rfc.fit(np.asarray(xTrain), np.asarray(yTrain))
ypred1 = rfc.predict(xTrain)
print('Random Forest Training Accuracy = ',100*(ypred1.shape[0]-np.sum(np.abs(ypred1-yTrain)))/ypred1.shape[0])

score_c = clf.score(xTest,yTest) # 精确度
score_r = rfc.score(xTest,yTest)
print('Single Tree: {}'.format(score_c),'Random Forest: {}'.format(score_r))

# Testing DT
dump(clf, baseFolder1+'dt.joblib')
clf2 = load(baseFolder1+'dt.joblib')
ypred2 = clf2.predict(xTest)
print('Decision Tree Test Accuracy = ',100*(ypred2.shape[0]-np.sum(np.abs(ypred2-yTest)))/ypred2.shape[0])

# Testing RF
dump(rfc, baseFolder2+'rf.joblib')
rfc2 = load(baseFolder2+'rf.joblib')
ypred3 = rfc2.predict(xTest)
print('Random Forest Test Accuracy = ',100*(ypred3.shape[0]-np.sum(np.abs(ypred3-yTest)))/ypred3.shape[0])

