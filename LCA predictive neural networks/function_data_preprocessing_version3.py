# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:53:06 2018

@author: e0225113
"""


import numpy as np
from sklearn.model_selection import train_test_split




#%%
def data_preprocessing(x,y, seed1, seed2, seed3):
    

    
    # 將 feature standardized
    
    my_xscaler = StandardScaler()
    my_xscaler = my_xscaler.fit(x)
    x_std = my_xscaler.transform(x)
    
    my_yscaler = StandardScaler()
    my_yscaler = my_yscaler.fit(np.array([y]).reshape(-1,1))
    y_std = my_yscaler.transform(np.array([y]).reshape(-1,1))
    
    # PCA 
    #  File "C:\Users\e0225113\anaconda3\envs\my-rdkit-env\lib\site-packages\sklearn\decomposition\pca.py", line 87, in _assess_dimension_
#    (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

# 為了PCA裡面有個log(0)的問題 這邊把log(n_samples + 1) 讓她不會變成0
# 把
#            pa += log((spectrum[i] - spectrum[j]) *
#                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)
            
#            改成
            
#             pa += log((spectrum[i] - spectrum[j]) *
#                      (1. / spectrum_[j] - 1. / spectrum_[i])+ 1e-99 ) + log(n_samples)
    x_original = np.float64(x)
    x_std = x_std
    x_pca99 = get_data_treatedPCA(x_std, 0.99)
    x_pca80 = get_data_treatedPCA(x_std, 0.8)
    # x_log1p = np.log1p(x)   這邊不能如此做，因為會遇到bug RuntimeWarning: invalid value encountered in log1p
    # 原因在 如果使用
    
    xList = [x_original] +[x_std] + [x_pca99] + [x_pca80]
    
    
    print ('the shape of pca for 99 % is ', x_pca99.shape)
    print ('the shape of pca for 80 % is ', x_pca80.shape)
    
    
    y_original = y
    y_std = y_std
    y_log1p = np.log1p(y)
    y_log1p = np.float64(y_log1p)
    
    yList = [y_original] + [y_std] + [y_log1p]
    
    xName = ['original x', 'standardized x', 'PCA (99%) x', 'PCA (80%) x']
    yName = ['original y', 'standardized y', 'log1p y' ]
    nameList = []
    for x in xName:
        for y in yName:
            nameList += [x+', '+y]
    
    dataDict = {}
    nameList =[]
    for xData, xCheck in zip(xList, xName):
        for yData, yCheck in zip(yList, yName):
            nameList += [xCheck + ', '+ yCheck]
            print(xCheck + ', ' + yCheck)
            
            xTrain42, xTest42, yTrain42, yTest42 = train_test_split(xData, yData, test_size = 0.2, random_state = seed1)
            a = [xTrain42, xTest42, yTrain42, yTest42]
            xTrain300, xTest300, yTrain300, yTest300 = train_test_split(xData, yData, test_size = 0.2, random_state = seed2)
            b = [xTrain300, xTest300, yTrain300, yTest300]
            xTrain5000, xTest5000, yTrain5000, yTest5000 = train_test_split(xData, yData, test_size = 0.2, random_state = seed3)
            c = [xTrain5000, xTest5000, yTrain5000, yTest5000]
            
            dataDict[xCheck + ', ' + yCheck] = [a, b, c]
            
    return dataDict, nameList, my_yscaler
    
    
    
#    # 下面要用 x_std 以及 x_pca 分別去做，所以分別分開
#    
#    x_std_train, x_std_test, y_std_train, y_std_test = train_test_split(x_std, y, test_size = 0.2, random_state = 42)
#    
#    x_pca_train, x_pca_test, y_pca_train, y_pca_test = train_test_split(x_pca, y, test_size =0.2, random_state =42)