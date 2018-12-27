# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:19:28 2018

@author: e0225113
"""

#%%

import xgboost as xgb
import lightgbm as lgb




from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import norm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import os


#%%

# windows
os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\chemicals LCIA data\\準備用來預測的data')

# mac
#os.chdir('/Volumes/GoogleDrive/我的雲端硬碟/NUS/Paper in Wang\'s lab/papers/2018 11 4 LCA_AI 2/chemicals LCIA data/準備用來預測的data')


ei99_e = pd.read_csv('EI99 ecosystem vs descriptors.csv')
ei99_e = ei99_e.drop([ei99_e.columns[0], ei99_e.columns[1], ei99_e.columns[2], ei99_e.columns[3]], axis = 1)

ei99_h = pd.read_csv('EI99 health vs descriptors.csv')
ei99_h = ei99_h.drop([ei99_h.columns[0], ei99_h.columns[1], ei99_h.columns[2], ei99_h.columns[3]], axis = 1)

ei99_r = pd.read_csv('EI99 resourses vs descriptors.csv')
ei99_r = ei99_r.drop([ei99_r.columns[0], ei99_r.columns[1], ei99_r.columns[2], ei99_r.columns[3]], axis = 1)

ei99_total = pd.read_csv('EI99 total vs descriptors.csv')
ei99_total = ei99_total.drop([ei99_total.columns[0], ei99_total.columns[1], ei99_total.columns[2], ei99_total.columns[3]], axis = 1)

recipe_e = pd.read_csv('Recipe ecosystem vs descriptors.csv')
recipe_e = recipe_e.drop([recipe_e.columns[0], recipe_e.columns[1], recipe_e.columns[2], recipe_e.columns[3]], axis = 1)

recipe_h = pd.read_csv('Recipe health vs descriptors.csv')
recipe_h = recipe_h.drop([recipe_h.columns[0], recipe_h.columns[1], recipe_h.columns[2], recipe_h.columns[3]], axis = 1)

recipe_r = pd.read_csv('Recipe resources vs descriptors.csv')
recipe_r = recipe_r.drop([recipe_r.columns[0], recipe_r.columns[1], recipe_r.columns[2], recipe_r.columns[3]], axis = 1)

recipe_total = pd.read_csv('Recipe total vs descriptors.csv')
recipe_total = recipe_total.drop([recipe_total.columns[0], recipe_total.columns[1], recipe_total.columns[2], recipe_total.columns[3]], axis = 1)


dataList = [ei99_e, ei99_h, ei99_r, ei99_total, recipe_e, recipe_h, recipe_r, recipe_total]

#%%


# 因為best_ensemble_grid_search這個function跑出來的結果只會有5個，所以下面的resultList_ensemble我們設成5 columns
resultList_ensemble = np.ones(5).reshape(1,5)

for i, data in enumerate(dataList):
    print('_____________________________start_________________________________')
    print('_____________________________%s_________________________________' %i)
    
    #先把x , y分開
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    # 每組data都會得到5個結果
    predict_results, r2Test_results, mseTest_results, r2Train_results, mseTrain_results = best_ensemble_grid_search(x,y)
    #將這5個結果放到一個array裡面之後
    result = np.array([predict_results, r2Test_results, mseTest_results, r2Train_results, mseTrain_results]).reshape(1,5)
    # append到result list上面
    resultList_ensemble = np.append(resultList_ensemble, result, axis=0)
    print('=============================stop==================================')
    # 最後得到的result list會變成8 * 5   8組data 5個結果   
    # 每個cell裡面是不同data間，各種ensemble 的r2 or mse的比較




    
#%%
title_list = ['EI99 ecosystem', 'EI99 human health', 'EI99 resources', 'EI99 total', 'ReCiPe ecosystem', 'ReCiPe human health', 'ReCiPe resources', 'ReCiPe total']
fig_ensemble = make_resultList_into_graph_ensemble(result_list = resultList_ensemble, title_list = title_list)



# =============================================================================
# 這個是用來找每個data裡面的最大R2以及相對應的model
# =============================================================================
for row in range(1, len(resultList_ensemble)):
    print('%.4f' %min(resultList_ensemble[row, 2].values()))
    minn = min(resultList_ensemble[row, 2].values())
    list_of_keys = list(resultList_ensemble[row, 2].keys())
    list_of_values = list(resultList_ensemble[row,2].values())
    
    print(list_of_keys[list_of_values.index(minn)])
    
    
# mse 結果

#9.7688
#x: PCA, y: original, lightgbm
    
#3.0099
#x: PCA, y: standardized, xgboost
    
#1.7302
#x: PCA, y: log1p, xgboost
    
#33.0140
#x: PCA, y: original, xgboost
#    -----------------------------------------------------
#0.4264
#x: PCA, y: log1p, lightgbm
    
#3.7689
#x: PCA, y: standardized, xgboost
    
#3.3886
#x: standardized, y: log1p, xgboost
    
#18.9369
#x: PCA, y: original, xgboost
#    
    

# r2結果
    #EI99
#            0.0564
#            x: PCA, y: original, lightgbm
    
#            0.7610
#            x: PCA, y: standardized, xgboost
    
#            -0.0235
#            x: PCA, y: log1p, xgboost
    
#            0.4866
#            x: PCA, y: original, xgboost
    #recipe
#            0.0111
#            x: PCA, y: log1p, lightgbm
    
#            0.4609
#            x: PCA, y: standardized, xgboost
    
#            -0.0149
#            x: standardized, y: log1p, xgboost
    
#            0.2568
#            x: PCA, y: original, xgboost

