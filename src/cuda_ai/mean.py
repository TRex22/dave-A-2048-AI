# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:35:38 2017

@author: Liron
"""

import numpy as np
np.set_printoptions(threshold=np.nan)

data = np.genfromtxt("cuda_times.csv", delimiter=",", usecols=(0,1), max_rows=97, skip_header=96+96+96)
print data[0]
print data[data.shape[0]-1]

mean = np.zeros((16,2))

for i in np.arange(mean.shape[0]):
    x = data[6*i + 1:6*(i+1) + 1,1]
    mean[i,1] = np.mean(x)
    mean[i,0] = data[6*i + 1,0]
    
print mean

#np.savetxt("cuda_1000threads.txt", mean, delimiter=',')