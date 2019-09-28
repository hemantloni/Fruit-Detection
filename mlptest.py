import cv2
import os    
import numpy as np
import re
import glob
import sys
from NeuralNetworks import FunctionApproximator as mlp

def mlptest(path,name):
    
    image_np = cv2.imread(path)
    r_channel=0
    g_channel=0
    b_channel=0
    train=[]
    no=0
    
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            if image_np[i,j, 0]!=255 and image_np[i, j, 1]!=255 and image_np[i, j, 2]!=255:
                r_channel += image_np[i, j, 2]
                g_channel += image_np[i, j, 1]
                b_channel += image_np[i, j, 0]
                no += 1
                
    r_channel/=no
    g_channel/=no
    b_channel/=no
    
    r_channel/=255.
    g_channel/=255.
    b_channel/=255.
    
    
    dic={0.1:75 ,0.2:157 ,0.3:100 ,0.4:40 ,0.5:40 }
    

    
    
    train.append([name,r_channel,g_channel,b_channel])
    
    train = np.transpose(np.array(train))
    target = 56.35
    nn = mlp(train, target, 5, 1, 0.00001)
    
    nn.W1=np.load("W1.npy")
    nn.W2=np.load("W2.npy")
    nn.b1=np.load("b1.npy")
    nn.b2=np.load("b2.npy")

    v = 0
    if name != 0.6:
        weight = 0.2
        return 0.2 * nn.predict(train[:,v:v+1]) + 0.8 * dic[name] 
    return nn.predict(train[:,v:v+1])