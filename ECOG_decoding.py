#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 22:37:16 2021

@author: tianjinhua
"""
import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

fname = '/nfs/a2/userhome/tianjinhua/workingdir/nma/faceshouses.npz'
# Load data
alldat = np.load(fname, allow_pickle=True)['dat']

# select just one of the recordings here. 
dat1 = alldat[1][0] # passive view task
# dat2 = alldat[1][1]

# reorganize data: based on its current and last labels 
V = dat1['V'].astype('float32')

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

# make correlation matrix
index = 0
stimRDM = []
novelRDM = []
adaptRDM = []

# compute model RDM
for x in range(labelNum[0]):
    for y in range(labelNum[0]):
        if x != y and x + y < labelNum[0]: #x + y < 80:
            # novelRDM
            novelDiff = abs(novelLevel[x]-novelLevel[y])
            novelRDM.append(novelDiff)
            adaptDiff = abs(adaptLevel[x]-adaptLevel[y])
            adaptRDM.append(adaptDiff)
            if newLabel[x,0] == newLabel[y,1]:
                stimRDM.append(1)
            else:
                stimRDM.append(0)

# load neural RDM
RDMPath = '/nfs/a2/userhome/tianjinhua/workingdir/nma/ECOGRDM2x100BD_subj1.npy'
neuralRDM = np.load(RDMPath) # nTime,repeat,kfold,pairNum

# normalize RDM
nTime,repeat,kfold,RDM = neuralRDM.shape
neuralRDM = neuralRDM.reshape(nTime*repeat*kfold,RDM)
scaler = StandardScaler()
neuralRDM = scaler.fit_transform(neuralRDM)
neuralRDM = neuralRDM.reshape(nTime,repeat,kfold,RDM)

# make correlation matrix to record partial correlation results 
RDMcorrStim = np.zeros(nTime)
RDMpStim = np.zeros(nTime)
RDMcorrAdapt = np.zeros(nTime)
RDMpAdapt = np.zeros(nTime)
RDMcorrNovel = np.zeros(nTime)
RDMpNovel = np.zeros(nTime)

import pandas as pd
# calculate partial correlation between neural RDM and stiRDM, adaptRDM
for tp in range(nTime):
    datatmp = neuralRDM[tp,:] # subIndex,t,re,foldIndex,RDMindex 
    RDMtmp = np.average(neuralRDM[tp,:,:,:], axis=(0, 1))
    
    pdData = pd.DataFrame({'neuralRDM':RDMtmp,'stimRDM':stimRDM,'novelRDM':novelRDM,'adaptRDM':adaptRDM})

    corr=pg.partial_corr(pdData,x='neuralRDM',y='stimRDM',x_covar=['novelRDM','adaptRDM'],tail='two-sided',method='spearman') 
    RDMcorrStim[tp] = corr['r']
    RDMpStim[tp] = corr['p-val']

    corr=pg.partial_corr(pdData,x='neuralRDM',y='novelRDM',x_covar=['stimRDM','adaptRDM'],tail='two-sided',method='spearman') 
    RDMcorrNovel[tp] = corr['r']
    RDMpNovel[tp] = corr['p-val']
    
    corr=pg.partial_corr(pdData,x='neuralRDM',y='adaptRDM',x_covar=['stimRDM','novelRDM'],tail='two-sided',method='spearman') 
    RDMcorrAdapt[tp] = corr['r']
    RDMpAdapt[tp] = corr['p-val']

# plot time coure decoding results        
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6), dpi=300) #调用figure创建一个绘图对象
plt.plot(range(-20,nTime-20),RDMcorrStim,label='Stimuli type',color='brown')
plt.plot(range(-20,nTime-20),RDMcorrNovel,label='Novel Level',color='mediumblue')
plt.plot(range(-20,nTime-20),RDMcorrAdapt,label='Adapt Level',color='forestgreen')

threshold = 0.01
# plot significant p value 
RDMpStim[(RDMpStim>threshold)] = None
RDMpStim[(RDMpStim<=threshold)] = -0.68
RDMpNovel[(RDMpNovel>threshold)] = None
RDMpNovel[(RDMpNovel<=threshold)] = -0.71
RDMpAdapt[(RDMpAdapt>threshold)] = None
RDMpAdapt[(RDMpAdapt<=threshold)] = -0.74

plt.plot(range(-20,nTime-20),RDMpStim,color='brown')
plt.plot(range(-20,nTime-20),RDMpNovel,color='mediumblue')
plt.plot(range(-20,nTime-20),RDMpAdapt,color='forestgreen')# 'forestgreen'
plt.xlabel('Time points(10ms)')
plt.ylabel('Partial spearman correlation')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of correlations between MEG RDM and model RDMs') # partial Spearman
plt.legend()
plt.show()
