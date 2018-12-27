# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:50:11 2018

@author: e0225113
"""

#%%

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics.scorer import make_scorer


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]









#%%
# =============================================================================
# 這邊把best_nn_grid_search_version3得到的32個字典，放到np.array裡面，每一個row代表不同的data
# 每一個columns代表r2 or mse of both test and training data
# ei99 & recipe的 e h r t都被分開來跑 比較快
# =============================================================================

#%%

#下面跑的是第一組ei99
# 先用function_data_preprocessing，將data用三個不同的random seed分成不同的training test dataset後
# 放入function_best_nn_grid_search_version3 得到全部結果(32個字典包含R2 以及mse)
# 將上面得到的全部結果，放入np.array:    resultList_ei99裡面
# 再用function_make_resultList_into_graph_version4 畫出圖
# 最後收集各個group的best model 比較 用 function_graph_show_best_model_new  來畫出結果


#-----------------------------------------------------------
dataList_ei99_1 = [ei99_e]
dataListName_ei99_1 = ['ei99_e']
#-----------------------------------------------------------

my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_ei99 = np.ones(32).reshape(1,32)


#--------------------
for name, data in zip(dataListName_ei99_1, dataList_ei99_1):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
#------------------------------------
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    
# =============================================================================
# 下面只是用ei99_h來try，用來check dataset有沒有分成4種x 以及 3種y
# =============================================================================
# keys
'original x, original y'
'original x, standardized y'
'original x, log1p y'
'standardized x, original y'
'standardized x, standardized y'
'standardized x, log1p y'
'PCA (99%) x, original y'
'PCA (99%) x, standardized y'
'PCA (99%) x, log1p y'
'PCA (80%) x, original y'
'PCA (80%) x, standardized y'
'PCA (80%) x, log1p y'
    n=1
    for key in allData.keys():
        for i in range(len(allData[key])):
            for j in range(len(allData[key][i])):
                print(len(allData[key][i][j]))
                print('-------------------',n,'-----------')
                n = n+1
    # 最後 n 要等於 144   因為 4種x  *  3種y  *  = 12 種data process組合， 然後總共有3組split， 最後trainx testx trainy testy 共4種結果
    # 等於 12 * 3 *4 = 144
    
# =============================================================================
# 確定data preprocessing沒問題
# =============================================================================

    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_ei99 = np.append(resultList_ei99, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    
    
#----------------------------------------------------------
titleList_ei99 = ['EI99 ecosystem']
fig_relu_EI99_e, fig_elu_EI99_e,      fig_reluSTD_EI99_e, fig_eluSTD_EI99_e,        fig_reluPCA99_EI99_e,  fig_eluPCA99_EI99_e,        fig_reluPCA80_EI99_e, fig_eluPCA80_EI99_e = make_resultList_into_graph_version4(result_list = resultList_ei99, title_list =titleList_ei99)
#----------------------------------------------------------

#%%
# 將最好的model選出來，以及他們的mse，畫出比較圖
modelList_ei99e = ['2 Layer, 50 Neurons w/ relu\noriginal x, standardized y',                                                              '1 Layer, 50 Neurons w/ elu\noriginal x, original y',                                                                                                                                '1 Layer, 25 Neurons w/ relu\nstandardized x, original y',                                                                                                                  '2 Layer, 25 Neurons w/ elu\nstandardized x, log1p y',                                                                                                  '2 Layer, 25 Neurons w/ relu\nPCA (99%) x, standardized y',                                                                                                         '2 Layer, 100 Neurons w/ elu\nPCA (99%) x, standardized y',                                                                                                                     '3 Layer, 50 Neurons w/ relu\nPCA (80%) x, original y',                                                                                                                         '2 Layer, 25 Neurons w/ elu\nPCA (80%) x, original y',                                                                                                                                  'lightgbm\nPCA (90%) x, original y']

mseList_ei99e = [0.0704 , 0.0877, 0.8934, 0.1006, 0.0653, 0.0320, 0.0928, 0.0919, 9.7688]

graph_show_best_model_new(mseList = mseList_ei99e, modelList =modelList_ei99e, title='EI99 ecosystem')







#%%
# 下面跑的是第二組 ei99 human health

#-----------------------------------------------------------
dataList_ei99_2 = [ei99_h]
dataListName_ei99_2 = ['ei99_h']
#-----------------------------------------------------------

my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_ei99 = np.ones(32).reshape(1,32)


#--------------------
for name, data in zip(dataListName_ei99_2, dataList_ei99_2):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
#------------------------------------
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    

    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_ei99 = np.append(resultList_ei99, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    
    
#----------------------------------------------------------
titleList_ei99 = ['EI99 human health']
fig_relu_EI99_h, fig_elu_EI99_h,      fig_reluSTD_EI99_h, fig_eluSTD_EI99_h,        fig_reluPCA99_EI99_h,  fig_eluPCA99_EI99_h,        fig_reluPCA80_EI99_h, fig_eluPCA80_EI99_h = make_resultList_into_graph_version4(result_list = resultList_ei99, title_list =titleList_ei99)
#----------------------------------------------------------

#%%
modelList_ei99h = ['1 Layer, 250 Neurons w/ relu\noriginal x, standardized y',                                                                                                          '1 Layer, 100 Neurons w/ elu\noriginal x, log1p y',                                                                                                                                     '1 Layer, 100 Neurons w/ relu\nstandardized x, log1p y',                                                                                                                                    '2 Layer, 50 Neurons w/ elu\nstandardized x, original y',                                                                                                                                       '3 Layer, 50 Neurons w/ relu\nPCA (99%) x, original y',                                                                                                                             '2 Layer, 100 Neurons w/ elu\nPCA (99%) x, standardized y',                                                                                                                                 '3 Layer, 50 Neurons w/ relu\nPCA (80%) x, standardized y',                                                                                                                                                 '1 Layer, 250 Neurons w/ elu\nPCA (80%) x, log1p y',                                                                                                                                                        'xgboost\nPCA (99%) x, standardized y']

mseList_ei99h = [0.5019, 0.1599, 0.2158 ,0.0903, 0.3505, 0.1207 , 0.5340, 0.4185, 3.0099]


graph_show_best_model_new(mseList = mseList_ei99h, modelList =modelList_ei99h, title='EI99 human health')





#%%
# 下面跑的是第三組 ei99 resources

#-----------------------------------------------------------
dataList_ei99_3 = [ei99_r]
dataListName_ei99_3 = ['ei99_r']
#-----------------------------------------------------------

my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_ei99 = np.ones(32).reshape(1,32)


#--------------------
for name, data in zip(dataListName_ei99_3, dataList_ei99_3):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
#------------------------------------
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    

    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_ei99 = np.append(resultList_ei99, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    
    
#----------------------------------------------------------
titleList_ei99 = ['EI99 resources']
fig_relu_EI99_r, fig_elu_EI99_r,      fig_reluSTD_EI99_r, fig_eluSTD_EI99_r,        fig_reluPCA99_EI99_r,  fig_eluPCA99_EI99_r,        fig_reluPCA80_EI99_r, fig_eluPCA80_EI99_r = make_resultList_into_graph_version4(result_list = resultList_ei99, title_list =titleList_ei99)
#----------------------------------------------------------


#%%
modelList_ei99r = ['1 Layer, 50 Neurons w/ relu\noriginal x, standardized y',                                                                                                           '3 Layer, 50 Neurons w/ elu\noriginal x, standardized y',                                                                                                                       '3 Layer, 50 Neurons w/ relu\nstandardized x, log1p y',                                                                                                             '3 Layer, 25 Neurons w/ elu\nstandardized x, log1p y',                                                                                                                                          '2 Layer, 50 Neurons w/ relu\nPCA (99%) x, standardized y',                                                                                                     '3 Layer, 25 Neurons w/ elu\nPCA (99%) x, standardized y',                                                                                                                          '2 Layer, 50 Neurons w/ relu\nPCA (80%) x, log1p y',                                                                                                                                    '2 Layer, 50 Neurons w/ elu\nPCA (80%) x, standardized y',                                                                                                                  'xgboost\nPCA (99%) x, log1p y']

mseList_ei99r = [0.0450 , 0.0478, 0.1190, 0.0387, 0.0307, 0.0186, 0.2127, 0.1587, 1.7302]




graph_show_best_model_new(mseList = mseList_ei99r, modelList =modelList_ei99r, title='EI99 resources')








#%%
# 下面跑的是第四組 ei99 total

#-----------------------------------------------------------
dataList_ei99_4 = [ei99_total]
dataListName_ei99_4 = ['ei99_total']
#-----------------------------------------------------------

my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_ei99 = np.ones(32).reshape(1,32)


#--------------------
for name, data in zip(dataListName_ei99_4, dataList_ei99_4):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
#------------------------------------
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    

    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_ei99 = np.append(resultList_ei99, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    
    
#----------------------------------------------------------
titleList_ei99 = ['EI99 total']
fig_relu_EI99_t, fig_elu_EI99_t,      fig_reluSTD_EI99_t, fig_eluSTD_EI99_t,        fig_reluPCA99_EI99_t,  fig_eluPCA99_EI99_t,        fig_reluPCA80_EI99_t, fig_eluPCA80_EI99_t = make_resultList_into_graph_version4(result_list = resultList_ei99, title_list =titleList_ei99)
#----------------------------------------------------------

#%%
modelList_ei99t = ['1 Layer, 250 Neurons w/ relu\noriginal x, standardized y',                                                                                                                  '1 Layer, 50 Neurons w/ elu\noriginal x, original y',                                                                                                                                   '1 Layer, 25 Neurons w/ relu\nstandardized x, log1p y',                                                                                                                 '2 Layer, 50 Neurons w/ elu\nstandardized x, log1p y',                                                                                                                                  '3 Layer, 50 Neurons w/ relu\nPCA (99%) x, log1p y',                                                                                                                        '1 Layer, 25 Neurons w/ elu\nPCA (99%) x, log1p y',                                                                                                                                 '3 Layer, 50 Neurons w/ relu\nPCA (80%) x, log1p y',                                                                                                                            '2 Layer, 50 Neurons w/ elu\nPCA (80%) x, log1p y',                                                                                                                                             'xgboost\nPCA (99%) x, original y']

mseList_ei99t = [0.8306, 1.2456, 0.4715, 0.6562 , 0.9774, 0.3781, 1.5587, 1.9395, 33.0140]



graph_show_best_model_new(mseList = mseList_ei99t, modelList =modelList_ei99t, title='EI99 total')































#%%
# =============================================================================
# 這邊把best_nn_grid_search_version3得到的32個字典，放到np.array裡面，每一個row代表不同的data
# 每一個columns代表r2 or mse of both test and training data
# ei99 & recipe的 e h r t都被分開來跑 比較快
# =============================================================================


# 下面跑得是第一組recipe
dataList_recipe_1 = [recipe_e]
dataListName_recipe_1 = ['recipe_e']



my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_recipe = np.ones(32).reshape(1,32)

for name, data in zip(dataListName_recipe_1, dataList_recipe_1):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
    
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    


    
    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_recipe = np.append(resultList_recipe, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    #%%
titleList_recipe = ['ReCiPe ecosystem']
fig_relu_ReCiPe_e, fig_elu_ReCiPe_e,      fig_reluSTD_ReCiPe_e, fig_eluSTD_ReCiPe_e,        fig_reluPCA99_ReCiPe_e,  fig_eluPCA99_ReCiPe_e,        fig_reluPCA80_ReCiPe_e, fig_eluPCA80_ReCiPe_e = make_resultList_into_graph_version4(result_list = resultList_recipe, title_list =titleList_recipe)





#%%
modelList_recipe_e = ['1 Layer, 50 Neurons w/ relu\noriginal x, standardized y',                                                                                                                    '3 Layer, 25 Neurons w/ elu\noriginal x, standardized y',                                                                                                                       '2 Layer, 50 Neurons w/ relu\nstandardized x, standardized y',                                                                                                                          '2 Layer, 25 Neurons w/ elu\nstandardized x, log1p y',                                                                                                                          '3 Layer, 25 Neurons w/ relu\nPCA (99%) x, original y',                                                                                                                                     '3 Layer, 25 Neurons w/ elu\nPCA (99%) x, original y',                                                                                                          '3 Layer, 50 Neurons w/ relu\nPCA (80%) x, standardized y',                                                                                                                                 '2 Layer, 50 Neurons w/ elu\nPCA (80%) x, standardized y',                                                                                                                              'lightgbm, PCA (90%) x, log1p y']

mseList_recipe_e = [0.0303 , 0.0183 , 0.0515, 0.0153 , 0.0164 , 0.0228, 0.0220 , 0.0182, 0.4264]








graph_show_best_model_new(mseList = mseList_recipe_e, modelList =modelList_recipe_e, title='ReCiPe ecosystem')


#    #recipe
#    
#
#x: PCA, y: log1p, 



#%%





# 下面跑得是第二組recipe
dataList_recipe_2 = [recipe_h]
dataListName_recipe_2 = ['recipe_h']



my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_recipe = np.ones(32).reshape(1,32)

for name, data in zip(dataListName_recipe_2, dataList_recipe_2):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
    
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    
    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_recipe = np.append(resultList_recipe, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    

titleList_recipe = ['ReCiPe human health']
fig_relu_ReCiPe_h, fig_elu_ReCiPe_h,      fig_reluSTD_ReCiPe_h, fig_eluSTD_ReCiPe_h,        fig_reluPCA99_ReCiPe_h,  fig_eluPCA99_ReCiPe_h,        fig_reluPCA80_ReCiPe_h, fig_eluPCA80_ReCiPe_h = make_resultList_into_graph_version4(result_list = resultList_recipe, title_list =titleList_recipe)
#%%

modelList_recipe_h = ['2 Layer, 100 Neurons w/ relu\noriginal x, standardized y',                                                                                                           '1 Layer, 250 Neurons w/ elu\noriginal x, original y',                                                                                                                          '2 Layer, 50 Neurons w/ relu\nstandardized x, original y',                                                                                                                              '2 Layer, 25 Neurons w/ elu\nstandardized x, log1p y',                                                                                                                      '2 Layer, 25 Neurons w/ relu\nPCA (99%) x, standardized y',                                                                                                                                 '2 Layer, 25 Neurons w/ elu\nPCA (99%) x, log1p y',                                                                                                                 '2 Layer, 250 Neurons w/ relu\nPCA (80%) x, original y',                                                                                                                                        '3 Layer, 25 Neurons w/ elu\nPCA (80%) x, original y',                                                                                                                              'xgboost\n PCA (99%) x, standardized y']

mseList_recipe_h = [0.2077, 0.1855 , 0.0793, 0.1515, 0.1399, 0.0761, 0.2049 , 0.1921, 3.7689]

graph_show_best_model_new(mseList = mseList_recipe_h, modelList =modelList_recipe_h, title='ReCiPe human health')









#%%

# 下面跑得是第三組recipe
dataList_recipe_3 = [recipe_r]
dataListName_recipe_3 = ['recipe_r']



my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_recipe = np.ones(32).reshape(1,32)

for name, data in zip(dataListName_recipe_3, dataList_recipe_3):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
    
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    
    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_recipe = np.append(resultList_recipe, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
#%%
titleList_recipe = ['ReCiPe resources']
fig_relu_ReCiPe_r, fig_elu_ReCiPe_r,      fig_reluSTD_ReCiPe_r, fig_eluSTD_ReCiPe_r,        fig_reluPCA99_ReCiPe_r,  fig_eluPCA99_ReCiPe_r,        fig_reluPCA80_ReCiPe_r, fig_eluPCA80_ReCiPe_r = make_resultList_into_graph_version4(result_list = resultList_recipe, title_list =titleList_recipe)
#%%

modelList_recipe_r = ['3 Layer, 25 Neurons w/ relu\noriginal x, standardized y',                                                                                                                   '1 Layer, 250 Neurons w/ elu\noriginal x, standardized y',                                                                                                                      '1 Layer, 250 Neurons w/ relu\nstandardized x, standardized y',                                                                                                                 '2 Layer, 50 Neurons w/ elu\nstandardized x, log1p y',                                                                                                                                  '3 Layer, 25 Neurons w/ relu\nPCA (99%) x, standardized y',                                                                                                                         '2 Layer, 25 Neurons w/ elu\nPCA (99%) x, standardized y',                                                                                                                  '2 Layer, 100 Neurons w/ relu\nPCA (80%) x, standardized y',                                                                                                                        '2 Layer, 100 Neurons w/ elu\nPCA (80%) x, original y',                                                                                                                         'xgboost\nstandardized x, log1p y']


mseList_recipe_r = [0.2841 , 0.3597, 0.5104 , 0.1818 , 0.1969, 0.1128, 0.2996, 0.6750,  3.3886]


graph_show_best_model_new(mseList = mseList_recipe_r, modelList =modelList_recipe_r, title='ReCiPe resources')







#%%


# 下面跑得是第四組recipe
dataList_recipe_4 = [recipe_total]
dataListName_recipe_4 = ['recipe_total']



my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]
resultList_recipe = np.ones(32).reshape(1,32)

for name, data in zip(dataListName_recipe_4, dataList_recipe_4):
    print('#############################################################################################')
    print('------------------------------------data is %s---------------------------------------------'  %name)
    
    
    
    columns = data.columns
    x = data.drop([columns[0]], axis=1)
    y = data[columns[0]]
    
    allData, nameList, my_yscaler = data_preprocessing(x,y, 42, 135135, 67896789)
    
    allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu = best_nn_grid_search_version3(allData, layer_list = my_layer, neuron_list = my_neuron, my_yscaler=my_yscaler)
    
    
    
    
    
    result = np.array([allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu]).reshape(1,32)
    
    resultList_recipe = np.append(resultList_recipe, result, axis=0)
    print('______________stop__________________________stop_________________________stop___________________')
    
    
titleList_recipe = ['ReCiPe total']
fig_relu_ReCiPe_t, fig_elu_ReCiPe_t,      fig_reluSTD_ReCiPe_t, fig_eluSTD_ReCiPe_t,        fig_reluPCA99_ReCiPe_t,  fig_eluPCA99_ReCiPe_t,        fig_reluPCA80_ReCiPe_t, fig_eluPCA80_ReCiPe_t = make_resultList_into_graph_version4(result_list = resultList_recipe, title_list =titleList_recipe)
#%%

modelList_recipe_t = ['1 Layer, 100 Neurons w/ relu\noriginal x, original y',                                                                                                           '1 Layer, 50 Neurons w/ elu\noriginal x, standardized y',                                                                                                                       '3 Layer, 25 Neurons w/ relu\nstandardized x, log1p y',                                                                                                                                 '3 Layer, 25 Neurons w/ elu\nstandardized x, original y',                                                                                                                       '2 Layer, 50 Neurons w/ relu\nPCA (99%) x, original y',                                                                                                                                         '2 Layer, 50 Neurons w/ elu\nPCA (99%) x, standardized y',                                                                                                                              '2 Layer, 25 Neurons w/ relu\nPCA (80%) x, original y',                                                                                                                                         '2 Layer, 50 Neurons w/ elu\nPCA (80%) x, original y',                                                                                                                              'xgboost\n PCA (99%) x, original y']

mseList_recipe_t = [0.5099, 1.1813, 0.8580, 0.6675, 0.4660 , 0.4393, 1.4029, 1.1719, 18.9369]

graph_show_best_model_new(mseList = mseList_recipe_t, modelList =modelList_recipe_t, title='ReCiPe total')









#%%

# ensumble結果
    #EI99
#9.7688
#x: PCA, y: original, lightgbm
#3.0099
#x: PCA, y: standardized, xgboost
#1.7302
#x: PCA, y: log1p, xgboost
#33.0140
#x: PCA, y: original, xgboost
#
##---------------------------------
#    #recipe
#    
#0.4264
#x: PCA, y: log1p, lightgbm
#3.7689
#x: PCA, y: standardized, xgboost
#3.3886
#x: standardized, y: log1p, xgboost
#18.9369
#x: PCA, y: original, xgboost
    

