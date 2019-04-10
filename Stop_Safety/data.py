import os
import h5py
import numpy as np
import scipy.io as sio
import sys

def loadData(filedir):
    print("Loading from {}".format(filedir))
    files = os.listdir(filedir)

    images = []
    targets = []

    for i in range(len(files)):
        fp = files[i]
        fpfull = filedir + fp
        data = h5py.File(fpfull, "r")
        
        images.append(np.array(data['rgb']))
        targets.append(np.array(data['targets'][:,24] == 6.0).astype(int))
    
        if i % 50 == 0:
            print("Loaded {}/{} files".format(i, len(files)))
    
    X = np.concatenate(images, axis=0)
    y = np.concatenate(targets, axis=0).reshape((-1, 1))

    print(X.shape)
    print(y.shape)
    print(np.sum(y))

    return X, y

def convertOneHot(target):
    y_onehot = np.zeros((target.shape[0], 10))
    for item in range(0, target.shape[0]):
        y_onehot[item][target[item] - 1] = 1
    
    return y_onehot

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target