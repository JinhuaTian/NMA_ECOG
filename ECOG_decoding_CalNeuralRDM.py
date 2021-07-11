#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:02:39 2021

@author: tianjinhua
"""
# @title Data retrieval
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

index = 0
for i in range(windowLength,dat1['stim_id'].shape[0]):
    tmpLabel = dat1['stim_id'][(i-windowLength):i+1] 
    newLabel[index] = tmpLabel[::-1] # invert the label number
    index = index + 1

for i in range(newLabel.shape[0]):
    for n in range(newLabel.shape[1]):
        labelNum[i] = labelNum[i] + (2**n)*newLabel[i,n]

# count data labe and related sample number
countlabel = False # 12~21
if countlabel == True:        
    from collections import Counter
    print(Counter(labelNum))

# Calculate Neuron RDM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# compute pair number:
pairNum = 0
for x in range(labelNum[0]):
    for y in range(labelNum[0]):
        if x != y and x + y < labelNum[0]: # exclude same id
            pairNum = pairNum + 1
            
# construct model RMD: 1.stimuli type 2.novelty/adaptation level (0~3)
            
# prepare ECOG data #V_epochs: sample, timenumber, channel number
# reshampe data and reduce dimension using PCA
#resample data at each 10 time points
avgV_epochs = np.zeros([300,80,50])
for t in range(80):
    avgV_epochs[:,t,:] = np.average(V_epochs[:,t*10:t*10+10,:],axis=1)
nSample, nTime, nChan = avgV_epochs.shape
avgV_epochs = avgV_epochs.reshape(nSample*nTime, nChan)

# dimension reduction (99% explained variance)
pca = PCA(n_components=0.99, svd_solver="full")  # n_components=0.90,
pca = pca.fit(avgV_epochs)
Vpc = pca.transform(avgV_epochs)
print('PC number is '+ str(Vpc.shape[1]))
Vpc = Vpc.reshape(nSample, nTime, Vpc.shape[1])
Vpc = Vpc[windowLength:,:,:] # start from the windowlength po
# traububg data 
import time
scaler = StandardScaler()
repeat = 100
kfold = 2
 
accs = np.zeros([nTime,repeat,kfold,pairNum])
# RDM of decoding accuracy values for each time point
for t in range(nTime): # notice that decoding start windowLength 
    # pick the time data and normalize data
    Xt = Vpc[:,t,:]
    Xt = scaler.fit_transform(Xt)

    time0 = time.time()
    #repeat for repeat times:
    for re in range(repeat):
        state = np.random.randint(0,100)
        kf=StratifiedKFold(n_splits=kfold, shuffle=True,random_state=state)
        foldIndex = 0
        for train_index, test_index in kf.split(Xt,labelNum):
            xTrain, xTest, yTrain, yTest, = Xt[train_index], Xt[test_index],labelNum[train_index],labelNum[test_index]
            yTrain = yTrain.reshape(yTrain.shape[0],1)
            yTest = yTest.reshape(yTest.shape[0],1)
            trainPd = np.concatenate((yTrain,xTrain),axis=1) # train data
            testPd = np.concatenate((yTest,xTest),axis=1) # test data
            RDMindex = 0
            for x in range(labelNum[0]):
                for y in range(labelNum[0]):
                    if x != y and x + y < labelNum[0]: #x + y < 82:
                        Pd1 = trainPd[(trainPd[:,0] == (x+1)) | (trainPd[:,0] == (y+1))] # labels are 1~80
                        Pd2 = testPd[(testPd[:,0] == (x+1)) | (testPd[:,0] == (y+1))]
                        # run svm
                        svm = SVC(kernel="linear")
                        svm.fit(Pd1[:,1:],Pd1[:,0])
                        acc = svm.score(Pd2[:,1:],Pd2[:,0]) # sub,time,RDMindex,fold,repeat # subIndex,t,re,foldIndex,RDMindex
                        # save acc
                        accs[t,re,foldIndex,RDMindex]=acc
                        RDMindex = RDMindex + 1
            foldIndex = foldIndex + 1
    time_elapsed = time.time() - time0
    print('Time point {} finished in {:.0f}m {:.0f}s'.format(t, time_elapsed // 60, time_elapsed % 60)) # + 'repeat '+ str(re)

# save neural RDM 
np.save("/nfs/a2/userhome/tianjinhua/workingdir/nma/ECOGRDM2x100_subj1.npy",accs)

partialAvgAcc = np.average(accs, axis=(1, 2, 3))
import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-20,60),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')