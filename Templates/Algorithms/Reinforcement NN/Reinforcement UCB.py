#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:27:10 2018

@author: studyrelated
"""
"""
Upper bound confidence algorithm on nxn matrix of 10 ads per person. 
"""
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing the UCB Algorithm
#Global variables
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
reward = 0
total_reward = 0

#Average reward of ad i up to round n
for n in range(0,N) :
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i] >0):
            #Calculating Average reward of i
            average_reward = sum_of_rewards[i]/numbers_of_selections[i]
            #Delta function
            delta_i = math.sqrt(3/2* math.log(n+1)/numbers_of_selections[i])
            #Calculating Upper Bound
            upper_bound = average_reward + delta_i
            #High upper bound to iterate over next i
        else:
            upper_bound=1e400
        #Keeping track of index of specific ad
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    
    #Add number of times ad is selected in list 
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
            
    #Real reward to update sums_of_rewards
    reward = dataset.values[n,ad]  
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward      
            
    #Update total_reward
    total_reward = total_reward + reward        
            
#Visualising results
plt.hist(ads_selected)
plt.title('Histogram of ad selections')
plt.xlabel('Ad')
plt.ylabel('Number of ad selection')
plt.show()       
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
