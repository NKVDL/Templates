#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 02:31:13 2018

@author: studyrelated
"""
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
#Global variables
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

reward = 0
total_reward = 0

#Average reward of ad i up to round n
for n in range(0,N) :
    ad = 0
    max_random = 0
    for i in range(0,d):
        #random draws from bitoulli distrubition
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] +1)
        #Keeping track of index of specific ad
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    
    #Add number of times ad is selected in list 
    ads_selected.append(ad)
    #Reward of i == 1 or 0
    reward = dataset.values[n,ad] 
    #Increment vector with 1 if ad is selected
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] +1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] +1
        
    #Update total_reward
    total_reward = total_reward + reward        
            
#Visualising results
plt.hist(ads_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ad')
plt.ylabel('Number of ad selection')
plt.show()       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            