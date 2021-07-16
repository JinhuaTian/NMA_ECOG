#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:53:27 2021

@author: tianjinhua
"""
import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as LR
import statsmodels.api as sm

fname = '/nfs/a2/userhome/tianjinhua/workingdir/nma/faceshouses.npz'
# Load data
alldat = np.load(fname, allow_pickle=True)['dat']

# select just one of the recordings here. 
dat1 = alldat[1][0] # passive view task
# dat2 = alldat[1][1]

# reorganize data: based on its current and last labels 
V = dat1['V'].astype('float32')

from scipy import signal
singalTransfer = True
# transfer raw data to boardband signal
if singalTransfer == True:
    b, a = signal.butter(3, [50], btype = 'high', fs=1000)
    V = signal.filtfilt(b,a,V,0)
    V = np.abs(V)**2
    b, a = signal.butter(3, [10], btype = 'low', fs=1000)
    V = signal.filtfilt(b,a,V,0)
    V = V/V.mean(0)

nt, nchan = V.shape
nstim = len(dat1['t_on'])

trange = np.arange(-200, 600)
ts = dat1['t_on'][:,np.newaxis] + trange
V_epochs = np.reshape(V[ts, :], (nstim, 800, nchan)) #sample, timenumber, channel number

# define new labels
dat1['stim_id'][dat1['stim_id']<=50] = 1
dat1['stim_id'][dat1['stim_id']>50] = 0

windowLength = 3
newLabel = np.zeros((dat1['stim_id'].shape[0]-windowLength,windowLength+1),dtype=int)
labelNum = np.zeros(dat1['stim_id'].shape[0]-windowLength,dtype=int)
novelLevel = np.zeros(dat1['stim_id'].shape[0]-windowLength,dtype=int)
adaptLevel = np.zeros(dat1['stim_id'].shape[0]-windowLength,dtype=int)

index = 0
for i in range(windowLength,dat1['stim_id'].shape[0]):
    tmpLabel = dat1['stim_id'][(i-windowLength):i+1] 
    newLabel[index] = tmpLabel[::-1] # invert the label number
    index = index + 1

# calculate model RDM
for i in range(newLabel.shape[0]):
    for n in range(newLabel.shape[1]):
        labelNum[i] = labelNum[i] + (2**n)*newLabel[i,n]
    if newLabel[i,0] != newLabel[i,1] and newLabel[i,1] == newLabel[i,2] and newLabel[i,2] == newLabel[i,3]:
        novelLevel[i] = 3
        adaptLevel[i] = 0
    elif newLabel[i,0] != newLabel[i,1] and newLabel[i,1] == newLabel[i,2] and newLabel[i,2] != newLabel[i,3]:
        novelLevel[i] = 2
        adaptLevel[i] = 0
    elif newLabel[i,0] != newLabel[i,1] and newLabel[i,1] != newLabel[i,2]:
        novelLevel[i] = 1
        adaptLevel[i] = 0
    elif newLabel[i,0] == newLabel[i,1] and newLabel[i,1] != newLabel[i,2] :
        novelLevel[i] = 0
        adaptLevel[i] = 1
    elif newLabel[i,0] == newLabel[i,1] and newLabel[i,1] == newLabel[i,2] and newLabel[i,2] != newLabel[i,3]:
        adaptLevel[i] = 2
        novelLevel[i] = 0
    elif newLabel[i,0] == newLabel[i,1] and newLabel[i,1] == newLabel[i,2] and newLabel[i,2] == newLabel[i,3]:
        adaptLevel[i] = 3
        novelLevel[i] = 0
        
# compute pair number:
pairNum = 0
for x in range(labelNum[0]):
    for y in range(labelNum[0]):
        if x != y and x + y < labelNum[0]: # exclude same id
            pairNum = pairNum + 1

# prepare ECOG data #V_epochs: sample, timenumber, channel number
# pick 43 and 46 channel
picks = [43,46]
#resample data at each 10 time points
avgV_epochs = np.zeros([300-windowLength,80,50])
# ignore the first windowLength epochs
V_epochs = V_epochs[windowLength:,:,:]
for t in range(80):
    avgV_epochs[:,t,:] = np.average(V_epochs[:,t*10:t*10+10,:],axis=1)
nSample, nTime, nChan = avgV_epochs.shape
avgV_epochs = avgV_epochs.reshape(nSample*nTime, nChan)
# normalize data
scaler = StandardScaler()
avgV_epochs = scaler.fit_transform(avgV_epochs)
avgV_epochs = avgV_epochs.reshape(nSample,nTime, nChan)

rMatrix = np.zeros([len(picks),nTime,3])
# run GLM or linear regression
# coef: newLabel[0,:], novelLevel
method = 'LR'
pickNum = 0
for pick in picks:
    pickEpochs = avgV_epochs[:,:,pick]
    for tps in range(nTime):
        pickTps = pickEpochs[:,tps,]
        x = np.concatenate((newLabel[:,0].reshape(297,1),novelLevel.reshape(297,1),adaptLevel.reshape(297,1)),axis = 1)
        if method == 'LR':
            linreg = LR()
            model = linreg.fit(x,pickTps)
            rMatrix[pickNum,tps,:] = linreg.coef_
        elif method == 'GLM':
            glmModel = sm.GLM(x,pickTps)
            glm_results = glmModel.fit() # glm_results.summary()
            param = glm_results.params
            rMatrix[pickNum,tps,:] = param
    pickNum = pickNum + 1
    
for pick in range(len(picks)):
    chanName = ' Channel ' + str(picks[pick])
    fig = plt.figure()
    plt.plot(np.arange(80),rMatrix[pick,:,0], label='Stimuli type', color='brown')
    plt.plot(np.arange(80),rMatrix[pick,:,1], label='Novel level', color='mediumblue')
    plt.plot(np.arange(80),rMatrix[pick,:,2], label='Adapt level', color='forestgreen')
    plt.xlabel('Time points(10ms)')
    plt.ylabel('Coefficient weights')
    plt.title('Time course of regression coefficient' + chanName)
    plt.legend()
    plt.show()    
