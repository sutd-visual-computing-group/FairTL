# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:39:20 2022

@author: Chris
"""
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch

#Preprocessing attribute list--------------------------------------------------------------
path="../../"
labelsDf=pd.read_csv(os.path.join(path, "CelebAMask-HQ-attribute-anno.csv"))
keys=list(labelsDf.keys())

#Variable to be adjusted to the SA
selectedKey=keys[21] #21=Gender
#Select Split mode (perc/min)
mode="all"

images=list(labelsDf[keys[0]])
filteredLabels=list(labelsDf[selectedKey])
print ("Orignal labelRatio=[%f,%f] with sampleCount=[%i,%i]"%(len(np.where(np.array(filteredLabels)==0)[0])/len(filteredLabels)
                                                              ,len(np.where(np.array(filteredLabels)==1)[0])/len(filteredLabels)
                                                              ,len(np.where(np.array(filteredLabels)==0)[0])
                                                              ,len(np.where(np.array(filteredLabels)==1)[0])))  
#maxPerLabel=min(len(np.where(np.array(filteredLabels)==0)[0]),len(np.where(np.array(filteredLabels)==1)[0]))


totalSamples=30000#50000 #CelebA-HQ 
label0=np.where(np.array(filteredLabels)==0)[0]
label1=np.where(np.array(filteredLabels)==1)[0]

numDref=min(len(label0),len(label1))
# label0=np.where(np.array(filteredLabels)==0)[0][len(label0)-int(numDref/2):len(label0)]
# label1=np.where(np.array(filteredLabels)==1)[0][len(label1)-int(numDref/2):len(label1)]
label0=np.where(np.array(filteredLabels)==0)[0][len(label0)-int(numDref):len(label0)]
label1=np.where(np.array(filteredLabels)==1)[0][len(label1)-int(numDref):len(label1)]
#Select images by index
index=np.concatenate((label0,label1))
#Load images
images=np.array([int(i.replace(".jpg","")) for i in images])[index]
filteredLabels=np.array(filteredLabels)[index]

    

#Preprocessing data list from file--------------------------------------------------------------
dataPath="../../celebA-HQ/data256x256"

outPathPrefix="../../Data"

dataList=np.array(os.listdir(dataPath))[images]
outImages=[]
#Opening the samples by index
for i in dataList:
    im = np.array(Image.open(os.path.join(dataPath, i)))
    outImages.append(im)
    
    
outImages=np.array(outImages) #Samples (need to process between -1 to 1)
# outLabels=filteredLabels #Labels 

outPathData=os.path.join(outPathPrefix,"CelebAHQ_FIDref_%s_data"%(selectedKey))
# outPathLabels=os.path.join(outPathPrefix,"CelebAHQ_%f_%s_Labels.pt"%(perc,selectedKey))
np.savez(outPathData, x=torch.from_numpy(np.moveaxis(outImages,3,1)))
# np.savez(torch.from_numpy(outLabels), outPathLabels)
    

