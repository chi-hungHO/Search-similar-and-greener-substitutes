# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:13:02 2018

@author: e0225113
"""

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
from sklearn.decomposition import PCA


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('ggplot')
import os







#%%
# windows
os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\chemicals LCIA data\\準備用來預測的data_剔除outlier')

# mac
#os.chdir('/Volumes/GoogleDrive/我的雲端硬碟/NUS/Paper in Wang\'s lab/papers/2018 11 4 LCA_AI 2/chemicals LCIA data/準備用來預測的data')


ei99_e = pd.read_csv('EI99 ecosystem vs descriptors_noOutlier.csv')
ei99_e = ei99_e.drop([ei99_e.columns[0], ei99_e.columns[1], ei99_e.columns[2], ei99_e.columns[3]], axis = 1)

ei99_h = pd.read_csv('EI99 health vs descriptors_noOutlier.csv')
ei99_h = ei99_h.drop([ei99_h.columns[0], ei99_h.columns[1], ei99_h.columns[2], ei99_h.columns[3]], axis = 1)

ei99_r = pd.read_csv('EI99 resourses vs descriptors_noOutlier.csv')
ei99_r = ei99_r.drop([ei99_r.columns[0], ei99_r.columns[1], ei99_r.columns[2], ei99_r.columns[3]], axis = 1)

ei99_total = pd.read_csv('EI99 total vs descriptors_noOutlier.csv')
ei99_total = ei99_total.drop([ei99_total.columns[0], ei99_total.columns[1], ei99_total.columns[2], ei99_total.columns[3]], axis = 1)

recipe_e = pd.read_csv('Recipe ecosystem vs descriptors_noOutlier.csv')
recipe_e = recipe_e.drop([recipe_e.columns[0], recipe_e.columns[1], recipe_e.columns[2], recipe_e.columns[3]], axis = 1)

recipe_h = pd.read_csv('Recipe health vs descriptors_noOutlier.csv')
recipe_h = recipe_h.drop([recipe_h.columns[0], recipe_h.columns[1], recipe_h.columns[2], recipe_h.columns[3]], axis = 1)

recipe_r = pd.read_csv('Recipe resources vs descriptors_noOutlier.csv')
recipe_r = recipe_r.drop([recipe_r.columns[0], recipe_r.columns[1], recipe_r.columns[2], recipe_r.columns[3]], axis = 1)

recipe_total = pd.read_csv('Recipe total vs descriptors_noOutlier.csv')
recipe_total = recipe_total.drop([recipe_total.columns[0], recipe_total.columns[1], recipe_total.columns[2], recipe_total.columns[3]], axis = 1)















'''以下使用ei99 4個category裡面分別最好的model 用test set看效果如何'''
#%%

# =============================================================================
# for EI99  E
# =============================================================================
x_ei99E = ei99_e.drop(ei99_e.columns[0], axis=1)
y_ei99E = ei99_e[ei99_e.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict EI99 e
EI99E_trainY, EI99E_yPred_train, EI99E_testY, EI99E_yPred = get_prediction_fr_best_ei99E(x_ei99E, y_ei99E)
#%% 畫出 整體 散佈圖
EI99Etotal_mse, EI99Etotal_mape, EI99Emse, EI99Emape = get_mape_mse_n_scatter( x_ei99E, y_ei99E, EI99E_trainY, EI99E_yPred_train, EI99E_testY, EI99E_yPred, title = 'EI99 ecosystem')
#%% 畫出 局部 散佈圖
EI99Etotal_mse, EI99Etotal_mape, EI99Emse, EI99Emape = get_mape_mse_n_scatter_selectedRange( x_ei99E, y_ei99E, EI99E_trainY, EI99E_yPred_train, EI99E_testY, EI99E_yPred, title = 'EI99 ecosystem (zoom in)')


#-------------- mse on test data:  [[2.92800408e-06]
# [4.92170103e-03]
# [1.07130866e-01]
# [2.03807790e-03]
# [2.48197027e-03]
# [2.81940989e-01]
# [3.90233759e-03]
# [7.87557159e-04]
# [6.42173164e-04]
# [1.56727847e-03]
# [3.62441059e-02]
# [3.42984244e-02]
# [5.91266209e-04]
# [2.55053243e-03]
# [3.80074638e-03]
# [1.75192682e-03]
# [3.97498422e-03]
# [1.70407685e-03]
# [5.30578000e-02]
# [1.28601946e-01]
# [7.00791364e-04]
# [8.25150210e-03]]
#
#_____ total mse:  0.03095199915492771
#
#_____ total mape:  8328.891209012783


#%%
# =============================================================================
# for EI99 H
# =============================================================================
x_ei99H = ei99_h.drop(ei99_h.columns[0], axis=1)
y_ei99H = ei99_h[ei99_h.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict EI99 h
EI99H_trainY, EI99H_yPred_train, EI99H_testY, EI99H_yPred = get_prediction_fr_best_ei99H(x_ei99H, y_ei99H)
#%% 畫出 整體 散佈圖
EI99Htotal_mse, EI99Htotal_mape, EI99Hmse, EI99Hmape = get_mape_mse_n_scatter( x_ei99H, y_ei99H, EI99H_trainY, EI99H_yPred_train, EI99H_testY, EI99H_yPred, title = 'EI99 human health')
#%% 畫出 局部 散佈圖
EI99Htotal_mse, EI99Htotal_mape, EI99Hmse, EI99Hmape = get_mape_mse_n_scatter_selectedRange( x_ei99H, y_ei99H, EI99H_trainY, EI99H_yPred_train, EI99H_testY, EI99H_yPred, title = 'EI99 human health (zoom in)')



#
#-------------- mse on test data:  [[3.07409253e-03]
# [3.95206133e-02]
# [1.02908288e+00]
# [2.27269999e-02]
# [1.17514297e-04]
# [7.95978265e+00]
# [9.55113385e-02]
# [1.74585422e-02]
# [7.93160069e-03]
# [4.87490441e-02]
# [2.67800786e-01]
# [5.58584884e-01]
# [1.63675672e-03]
# [3.12541116e-02]
# [3.19852040e-01]
# [5.20368967e-02]
# [9.52305871e-01]
# [1.74400089e-03]
# [2.58428697e-01]
# [5.03022548e-02]
# [4.94005720e-03]
# [3.43188521e-01]]
#
#
#_____ total mse:  0.5484559158635515
#
#_____ total mape:  686.7615460934089

#%%
# =============================================================================
# for EI99 R
# =============================================================================
x_ei99R = ei99_r.drop(ei99_r.columns[0], axis=1)
y_ei99R = ei99_r[ei99_r.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict EI99 r
EI99R_trainY, EI99R_yPred_train, EI99R_testY, EI99R_yPred = get_prediction_fr_best_ei99R(x_ei99R, y_ei99R)
#%% 畫出 整體 散佈圖
EI99Rtotal_mse, EI99Rtotal_mape, EI99Rmse, EI99Rmape = get_mape_mse_n_scatter( x_ei99R, y_ei99R, EI99R_trainY, EI99R_yPred_train, EI99R_testY, EI99R_yPred, title = 'EI99 resources')
#%% 畫出 局部 散佈圖
EI99Rtotal_mse, EI99Rtotal_mape, EI99Rmse, EI99Rmape = get_mape_mse_n_scatter_selectedRange( x_ei99R, y_ei99R, EI99R_trainY, EI99R_yPred_train, EI99R_testY, EI99R_yPred, title = 'EI99 resources (zoom in)')
#


#-------------- mse on test data:  [[2.40962619e-03]
# [3.51458524e-02]
# [5.74869200e-02]
# [2.28364392e-03]
# [5.99837098e-03]
# [1.15328146e-01]
# [6.10562612e-05]
# [7.32663355e-03]
# [1.37542148e-02]
# [2.45405333e-03]
# [3.88363091e-02]
# [7.92772625e-04]
# [1.17120108e-04]
# [1.80502068e-03]
# [2.77096871e-02]
# [4.06769562e-04]
# [1.46474480e-02]
# [4.04521910e-04]
# [3.37029803e-01]
# [9.06107580e-03]
# [1.56363344e-02]
# [2.92884253e-01]]
#
#_____ total mse:  0.044617256026574466
#
#_____ total mape:  3347.3378997444643


#%%
# =============================================================================
# for EI99 T
# =============================================================================

x_ei99T = ei99_total.drop(ei99_total.columns[0], axis=1)
y_ei99T = ei99_total[ei99_total.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict EI99 e
EI99T_trainY, EI99T_yPred_train, EI99T_testY, EI99T_yPred = get_prediction_fr_best_ei99T(x_ei99T, y_ei99T)
#%% 畫出 整體 散佈圖
EI99Ttotal_mse, EI99Ttotal_mape, EI99Tmse, EI99Tmape = get_mape_mse_n_scatter( x_ei99T, y_ei99T, EI99T_trainY, EI99T_yPred_train, EI99T_testY, EI99T_yPred, title = 'EI99 total')
#%% 畫出 局部 散佈圖
EI99Ttotal_mse, EI99Ttotal_mape, EI99Tmse, EI99Tmape = get_mape_mse_n_scatter_selectedRange( x_ei99T, y_ei99T, EI99T_trainY, EI99T_yPred_train, EI99T_testY, EI99T_yPred, title = 'EI99 total (zoom in)')



#-------------- mse on test data:  [[5.60083054e-03]
# [1.04380034e-01]
# [4.25096482e-01]
# [1.83164939e-01]
# [1.03270747e-01]
# [1.43056889e+01]
# [4.64305393e-02]
# [3.27802403e-03]
# [6.15682788e-02]
# [2.31658891e-01]
# [8.28535482e-03]
# [3.38637710e-01]
# [3.50663036e-01]
# [3.24535295e-02]
# [1.83063999e-01]
# [2.50080340e-02]
# [3.40638399e-01]
# [3.51258963e-02]
# [1.36437073e-01]
# [1.66734472e-01]
# [2.44295485e-02]
# [2.07117009e+00]]
#
#_____ total mse:  0.8719448
#
#_____ total mape:  335.1229234175249







#%%
# =============================================================================
# 以下是用來compare ei99的兩個total:  total predicted & E + R + H
# =============================================================================
EI99total_sum = EI99E_yPred + EI99H_yPred + EI99R_yPred

EI99Ttotal_mse, EI99Ttotal_mape, EI99Tmse, EI99Tmape = compare2_total( EI99total_sum, EI99T_yPred , EI99T_testY, title = 'EI99 total predicted vs E+R+H')




#------------------------- mse on EHR:  [[1.12876985e-02]
# [2.08323792e-01]
# [2.50117397e+00]
# [5.93835153e-02]
# [1.90740786e-02]
# [1.36300497e+01]
# [1.43892497e-01]
# [6.04127198e-02]
# [8.28122757e-06]
# [9.60498750e-02]
# [8.18921626e-01]
# [9.23021078e-01]
# [2.91067129e-03]
# [7.27794394e-02]
# [6.29907429e-01]
# [4.26127128e-02]
# [1.34545088e+00]
# [1.06408373e-02]
# [1.74040544e+00]
# [4.59795922e-01]
# [4.91967127e-02]
# [1.48315501e+00]]
#
#---------- mse on predicted:  [[5.60083054e-03]
# [1.04380034e-01]
# [4.25096482e-01]
# [1.83164939e-01]
# [1.03270747e-01]
# [1.43056889e+01]
# [4.64305393e-02]
# [3.27802403e-03]
# [6.15682788e-02]
# [2.31658891e-01]
# [8.28535482e-03]
# [3.38637710e-01]
# [3.50663036e-01]
# [3.24535295e-02]
# [1.83063999e-01]
# [2.50080340e-02]
# [3.40638399e-01]
# [3.51258963e-02]
# [1.36437073e-01]
# [1.66734472e-01]
# [2.44295485e-02]
# [2.07117009e+00]]
#
#_____EHR mse:  1.1049298
#_____predicted mse:  0.8719448 




# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================











































'''以下是用來算使用recipe 4個category裡面分別最好的model 用test set看效果如何'''

#%%

# =============================================================================
# for ReCiPe E
# =============================================================================

x_recipeE = recipe_e.drop(recipe_e.columns[0], axis=1)
y_recipeE = recipe_e[recipe_e.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict recipe e
recipeE_trainY, recipeE_yPred_train, recipeE_testY, recipeE_yPred = get_prediction_fr_best_recipeE(x_recipeE, y_recipeE)
#%% 畫出 整體 散佈圖
recipeEtotal_mse, recipeEtotal_mape, recipeEmse, recipeEmape = get_mape_mse_n_scatter( x_recipeE,                                                                                           y_recipeE,                                                                                                                                                                                  recipeE_trainY,                                                                                                                         recipeE_yPred_train,                                                                                                                                    recipeE_testY,                                                                                                                                                          recipeE_yPred, title = 'ReCiPe ecosystem')

#%% 畫出 局部 散佈圖
recipeEtotal_mse, recipeEtotal_mape, recipeEmse, recipeEmape = get_mape_mse_n_scatter_selectedRange( x_recipeE,                                                                       y_recipeE,                                                                                                                                                                                recipeE_trainY,                                                                                                                         recipeE_yPred_train,                                                                                                                                                            recipeE_testY,                                                                                                                                                                  recipeE_yPred, title = 'ReCiPe ecosystem (zoom in)')


#-------------- mse on test data:  [[2.5729612e-01]
# [2.6741324e-03]
# [1.7832579e-03]
# [1.2820099e-04]
# [8.4508021e-05]
# [4.8778555e-04]
# [1.0526262e-02]
# [4.9610157e-02]
# [1.4627738e-02]
# [7.3127947e-03]
# [2.9699388e-03]
# [3.9201223e-05]
# [1.4622939e-02]
# [1.2563858e-03]
# [7.8886386e-04]
# [1.0866837e-02]
# [1.3325184e-02]
# [3.5910616e-03]
# [1.4184114e-03]
# [1.9791216e-01]
# [4.8254311e-02]
# [7.6653936e-04]]
#
#_____ total mse:  0.02910649
#
#_____ total mape:  569.5541728626598




#%%
# =============================================================================
# for ReCiPe H
# =============================================================================

x_recipeH = recipe_h.drop(recipe_h.columns[0], axis=1)
y_recipeH = recipe_h[recipe_h.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict recipe h
recipeH_trainY, recipeH_yPred_train, recipeH_testY, recipeH_yPred = get_prediction_fr_best_recipeH(x_recipeH, y_recipeH)
#%% 畫出 整體 散佈圖
recipeHtotal_mse, recipeHtotal_mape, recipeHmse, recipeHmape = get_mape_mse_n_scatter( x_recipeH,                                                                                           y_recipeH,                                                                                                                                                                                  recipeH_trainY,                                                                                                                         recipeH_yPred_train,                                                                                                                                    recipeH_testY,                                                                                                                                                          recipeH_yPred, title = 'ReCiPe human health')

#%% 畫出 局部 散佈圖
recipeHtotal_mse, recipeHtotal_mape, recipeHmse, recipeHmape = get_mape_mse_n_scatter_selectedRange( x_recipeH,                                                                       y_recipeH,                                                                                                                                                                                recipeH_trainY,                                                                                                                         recipeH_yPred_train,                                                                                                                                                            recipeH_testY,                                                                                                                                                                  recipeH_yPred, title = 'ReCiPe human health (zoom in)')


#-------------- mse on test data:  [[1.7048365e-02]
# [7.3603038e-03]
# [8.4423876e-01]
# [1.5065481e-01]
# [2.2994020e-04]
# [6.3670340e+00]
# [2.1182586e-01]
# [1.5061137e-02]
# [9.0955570e-02]
# [4.7427710e-02]
# [4.1454009e-04]
# [2.7321966e-02]
# [6.1001461e-02]
# [5.0338828e-03]
# [3.0339391e-03]
# [2.2109678e-02]
# [4.8485108e-02]
# [2.8658461e-02]
# [4.5144770e-05]
# [5.8227861e-01]
# [6.2196329e-03]
# [2.8547163e-03]]
#
#_____ total mse:  0.38814974
#
#_____ total mape:  508.6372722278942




#%%
# =============================================================================
# for ReCiPe R
# =============================================================================

x_recipeR = recipe_r.drop(recipe_r.columns[0], axis=1)
y_recipeR = recipe_r[recipe_r.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict recipe r
recipeR_trainY, recipeR_yPred_train, recipeR_testY, recipeR_yPred = get_prediction_fr_best_recipeR(x_recipeR, y_recipeR)
#%% 畫出 整體 散佈圖
recipeRtotal_mse, recipeRtotal_mape, recipeRmse, recipeRmape = get_mape_mse_n_scatter( x_recipeR,                                                                                           y_recipeR,                                                                                                                                                                                  recipeR_trainY,                                                                                                                         recipeR_yPred_train,                                                                                                                                    recipeR_testY,                                                                                                                                                          recipeR_yPred, title = 'ReCiPe resources')

#%% 畫出 局部 散佈圖
recipeRtotal_mse, recipeRtotal_mape, recipeRmse, recipeRmape = get_mape_mse_n_scatter_selectedRange( x_recipeR,                                                                       y_recipeR,                                                                                                                                                                                  recipeR_trainY,                                                                                                                         recipeR_yPred_train,                                                                                                                                    recipeR_testY,                                                                                                                                                          recipeR_yPred, title = 'ReCiPe resources (zoom in)')


#
#-------------- mse on test data:  [[6.21300427e-04]
# [7.02533008e+00]
# [2.91662668e-02]
# [3.72443921e-02]
# [4.22638502e-04]
# [2.15671464e-01]
# [4.23135527e-02]
# [1.28629486e-03]
# [9.67979389e-03]
# [1.46490121e-03]
# [1.13047415e-03]
# [2.20344111e-03]
# [8.35118005e-03]
# [7.84810232e-04]
# [2.32130157e-02]
# [4.19081479e-02]
# [3.47755268e-03]
# [4.14758692e-03]
# [4.02340380e-03]
# [4.91239325e-03]
# [8.96325818e-03]
# [2.32232670e-05]]
#
#_____ total mse:  0.3393790530896886
#
#_____ total mape:  264.55831322763083



#%%
# =============================================================================
# for ReCiPe T
# =============================================================================

x_recipeT = recipe_total.drop(recipe_total.columns[0], axis=1)
y_recipeT = recipe_total[recipe_total.columns[0]]
# 使用 function_get_prediction_fr_best_models 用最好的model structure and preprocessing method來predict recipe t
recipeT_trainY, recipeT_yPred_train, recipeT_testY, recipeT_yPred = get_prediction_fr_best_recipeT(x_recipeT, y_recipeT)

#%% 畫出 整體 散佈圖
recipeTtotal_mse, recipeTtotal_mape, recipeTmse, recipeTmape = get_mape_mse_n_scatter( x_recipeT,                                                                                           y_recipeT,                                                                                                                                                                                  recipeT_trainY,                                                                                                                         recipeT_yPred_train,                                                                                                                                    recipeT_testY,                                                                                                                                                          recipeT_yPred, title = 'ReCiPe total')

#%% 畫出 局部 散佈圖
recipeTtotal_mse, recipeTtotal_mape, recipeTmse, recipeTmape = get_mape_mse_n_scatter_selectedRange( x_recipeT,                                                                                           y_recipeT,                                                                                                                                                                                  recipeT_trainY,                                                                                                                         recipeT_yPred_train,                                                                                                                                    recipeT_testY,                                                                                                                                                          recipeT_yPred, title = 'ReCiPe total (zoom in)')



#-------------- mse on test data:  [[3.43870051e-01]
# [9.01659848e+00]
# [3.88322233e-02]
# [1.27774368e-02]
# [9.81200387e-04]
# [3.61233498e+00]
# [6.74648178e-01]
# [8.74038179e-01]
# [2.36688168e-01]
# [1.69259549e-01]
# [9.17305232e-02]
# [5.43112711e-01]
# [1.38924621e+00]
# [3.13447471e-02]
# [6.45940847e-01]
# [1.75262141e-02]
# [2.43382639e+00]
# [1.50967043e-01]
# [8.67407233e-03]
# [4.82552277e-01]
# [4.69014703e-01]
# [1.74613750e-02]]
#
#_____ total mse:  0.9664284341728965
#
#_____ total mape:  479.3430460044918






#%%
# =============================================================================
# 以下是用來compare recipe的兩個total:  total predicted & E + R + H
# =============================================================================
recipetotal_sum = recipeE_yPred + recipeH_yPred + recipeR_yPred

recipeTTotal_mseEHR, recipeTTotal_msePred, recipeTmseEHR, recipeTmsePred = compare2_total( recipetotal_sum, recipeT_yPred , recipeT_testY, title = 'ReCiPe total predicted vs E+R+H')


##
#------------------------- mse on EHR:  [[1.61282672e-01]
# [7.77315608e+00]
# [6.24529669e-01]
# [3.24681477e-01]
# [1.44280334e-05]
# [9.05882826e+00]
# [5.90662273e-01]
# [9.58473873e-02]
# [2.71357182e-01]
# [8.83414560e-03]
# [4.59133831e-03]
# [1.25645402e-02]
# [2.10951813e-01]
# [5.60979528e-05]
# [5.54725485e-02]
# [6.20826537e-02]
# [2.68073031e-02]
# [2.02148057e-03]
# [1.16232316e-02]
# [1.50766892e-01]
# [1.54612334e-01]
# [9.34044586e-04]]
#
#---------- mse on predicted:  [[3.43870051e-01]
# [9.01659848e+00]
# [3.88322233e-02]
# [1.27774368e-02]
# [9.81200387e-04]
# [3.61233498e+00]
# [6.74648178e-01]
# [8.74038179e-01]
# [2.36688168e-01]
# [1.69259549e-01]
# [9.17305232e-02]
# [5.43112711e-01]
# [1.38924621e+00]
# [3.13447471e-02]
# [6.45940847e-01]
# [1.75262141e-02]
# [2.43382639e+00]
# [1.50967043e-01]
# [8.67407233e-03]
# [4.82552277e-01]
# [4.69014703e-01]
# [1.74613750e-02]]
#
#_____EHR mse:  0.8909853569001556
#_____predicted mse:  0.9664284341728965















































































'''以下是用來預測30 similar chemicals的LCA  分成兩部分 EI99 total and Recipe EHR'''
#%%
# =============================================================================
# 首先import data of 30 similar chemicals
# =============================================================================

os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_30_similarPoints')
data_similarPoints = pd.read_csv('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_30_similarPoints\\chemical 9843 similarPoint_clean.csv', index_col =0)






#%%
# =============================================================================
# 用來預測30 similar chemical的 ei99 t 
# =============================================================================
x_ei99T = ei99_total.drop(ei99_total.columns[0], axis=1)
y_ei99T = ei99_total[ei99_total.columns[0]]

a = np.zeros(30).reshape(-1,1)
aa = [] # 用來檢視的
for i in range(1,11):
    modelEI99_T, similarPred_EI99T = similarChemicals_ei99T(x_ei99T, y_ei99T, data_similarPoints)
    aa += [similarPred_EI99T[0,0]]
    a = np.append(a, similarPred_EI99T, axis=1)
std_a = np.std(a[:, 1:], axis=1).reshape(-1,1)
mean_a = np.mean(a[:, 1:], axis=1).reshape(-1,1)

#%%
# 這邊預測出來的trifluoroacetic anhydride的ei99 t 為0.78963499
# 而用回溯法做出來的是0.90985937

# 30個chemical 個別平均值
array([[0.78963499],
       [0.920533  ],
       [1.03606724],
       [0.69188598],
       [0.99555057],
       [0.81932078],
       [0.66660314],
       [0.74267244],
       [0.68193455],
       [0.78219787],
       [0.72839177],
       [0.80174953],
       [0.51142057],
       [0.56011664],
       [0.66988601],
       [0.72928058],
       [0.91557987],
       [0.98248271],
       [0.92406524],
       [0.91244903],
       [0.61223778],
       [1.23939573],
       [1.43233753],
       [1.01913767],
       [0.716442  ],
       [1.77430842],
       [0.5238039 ],
       [0.68364645],
       [0.69329105],
       [1.18884539]])
    
    
# 30個chemical 個別std
array([[0.63257821],
       [0.38734849],
       [0.62273265],
       [0.38340764],
       [0.65202668],
       [0.65910989],
       [0.29337394],
       [0.50314342],
       [0.39903574],
       [0.50185799],
       [0.47307664],
       [0.64571249],
       [0.27824673],
       [0.55821286],
       [0.36677398],
       [0.37059927],
       [0.5028721 ],
       [0.63063025],
       [0.45387202],
       [0.58364351],
       [0.5405174 ],
       [0.85181185],
       [0.88949143],
       [0.70013846],
       [0.49100323],
       [1.70726069],
       [0.47007631],
       [0.62836195],
       [0.34000188],
       [0.83366569]])
#%%
# =============================================================================
# 用來預測30 similar chemical的 recipe e
# =============================================================================
x_recipeE = recipe_e.drop(recipe_e.columns[0], axis=1)
y_recipeE = recipe_e[recipe_e.columns[0]]





b = np.zeros(30).reshape(-1,1)
bb = []
for i in range(1,11):
    modelRecipe_E, similarPred_RecipeE = similarChemicals_recipeE( x = x_recipeE, y = y_recipeE, my_data = data_similarPoints)
    bb += [similarPred_RecipeE[0,0]]
    b = np.append(b, similarPred_RecipeE, axis=1)
    
std_b = np.std(b[:, 1:], axis=1).reshape(-1,1)
mean_b = np.mean(b[:, 1:], axis=1).reshape(-1,1)
#%%
# 這邊預測出來的trifluoroacetic anhydride的recipe e 為0.12384146
# 而用回溯法做出來的是0.204209122

# 30個chemical 個別平均值
array([[0.12384146],
       [0.16592869],
       [0.10739622],
       [0.12024998],
       [0.13116553],
       [0.11736567],
       [0.11812583],
       [0.1075936 ],
       [0.16908455],
       [0.10821686],
       [0.11951891],
       [0.1906549 ],
       [0.12208899],
       [0.11013089],
       [0.15533957],
       [0.11302642],
       [0.10409935],
       [0.10920902],
       [0.13735361],
       [0.1492161 ],
       [0.19694137],
       [0.11423208],
       [0.1123167 ],
       [0.13267428],
       [0.11405928],
       [0.12871225],
       [0.10653447],
       [0.13223831],
       [0.11250321],
       [0.10617547]])
    
    
# 30個chemical 個別std
array([[0.09083551],
       [0.09382343],
       [0.08719754],
       [0.07135949],
       [0.07313319],
       [0.08698588],
       [0.07208068],
       [0.08406038],
       [0.09883105],
       [0.08593025],
       [0.07610456],
       [0.08916764],
       [0.07650735],
       [0.08685618],
       [0.08915754],
       [0.06672745],
       [0.08687744],
       [0.09152217],
       [0.07817978],
       [0.10746456],
       [0.10499506],
       [0.08128707],
       [0.07590357],
       [0.07704328],
       [0.07390245],
       [0.07614294],
       [0.0869565 ],
       [0.09709801],
       [0.09283065],
       [0.0843537 ]])
#%%
# =============================================================================
# 用來預測30 similar chemical的 recipe h
# =============================================================================
x_recipeH = recipe_h.drop(recipe_h.columns[0], axis=1)
y_recipeH = recipe_h[recipe_h.columns[0]]



c = np.zeros(30).reshape(-1,1)
cc = []
for i in range(1,11):
    modelRecipe_H, similarPred_RecipeH = similarChemicals_recipeH( x = x_recipeH, y = y_recipeH, my_data = data_similarPoints)       
    cc += [similarPred_RecipeH[0,0]]
    c = np.append(c, similarPred_RecipeH, axis=1)

std_c = np.std(c[:, 1:], axis=1).reshape(-1,1)
mean_c = np.mean(c[:, 1:], axis=1).reshape(-1,1)





#%%
# 這邊預測出來的trifluoroacetic anhydride的recipe h 為0.40653767
# 而用回溯法做出來的是0.526866971


# 30個chemical 個別平均值
array([[0.40653767],
       [0.45400009],
       [0.37425516],
       [0.45967307],
       [0.46628751],
       [0.42914638],
       [0.43996507],
       [0.42478137],
       [0.45511869],
       [0.43539783],
       [0.46111922],
       [0.55484543],
       [0.44995688],
       [0.32301374],
       [0.41257298],
       [0.43673699],
       [0.44044997],
       [0.42329974],
       [0.43468915],
       [0.40022874],
       [0.35648188],
       [0.48073199],
       [0.56078485],
       [0.48674197],
       [0.45891526],
       [0.63278411],
       [0.30862764],
       [0.36267911],
       [0.42831258],
       [0.45654748]])
    
    
# 30個chemical 個別std
array([[0.23879644],
       [0.16240324],
       [0.27359668],
       [0.23635181],
       [0.2746625 ],
       [0.29707615],
       [0.20623059],
       [0.259135  ],
       [0.14392162],
       [0.28520345],
       [0.26542739],
       [0.30541237],
       [0.18093478],
       [0.16879939],
       [0.09994544],
       [0.20654688],
       [0.30586283],
       [0.30614554],
       [0.29795344],
       [0.23019077],
       [0.17756472],
       [0.33340689],
       [0.41023392],
       [0.25625159],
       [0.24714081],
       [0.74540097],
       [0.16292471],
       [0.16473397],
       [0.20827226],
       [0.29905446]])
#%%
# =============================================================================
# 用來預測30 similar chemical的 recipe r
# =============================================================================
x_recipeR = recipe_r.drop(recipe_r.columns[0], axis=1)
y_recipeR = recipe_r[recipe_r.columns[0]]




d = np.zeros(30).reshape(-1,1)
dd = []
for i in range(1,11):
    modelRecipe_R, similarPred_RecipeR = similarChemicals_recipeR( x = x_recipeR, y = y_recipeR, my_data = data_similarPoints)
    dd += [similarPred_RecipeR[0,0]]
    d = np.append(d, similarPred_RecipeR, axis=1)

std_d = np.std(d[:, 1:], axis=1).reshape(-1,1)
mean_d = np.mean(d[:, 1:], axis=1).reshape(-1,1)

#%%
## 這邊預測出來的trifluoroacetic anhydride的recipe r 為0.40653767
## 而用回溯法做出來的是0.447340286
#
# 30個chemical 個別平均值
array([[0.41672769],
       [0.51410746],
       [0.28975599],
       [0.53919222],
       [0.91934793],
       [0.54710141],
       [0.42049284],
       [0.44214268],
       [0.51321385],
       [0.44808122],
       [0.57938132],
       [0.49527258],
       [0.35469955],
       [0.41505996],
       [0.53957101],
       [0.43791603],
       [0.42919855],
       [0.57192364],
       [0.47782753],
       [0.56463989],
       [0.40377189],
       [0.86672682],
       [0.5441097 ],
       [0.38731678],
       [0.60268539],
       [0.83272568],
       [0.35107063],
       [0.47500924],
       [0.52669874],
       [0.82000087]])
    
    
# 30個chemical 個別std
array([[0.2971404 ],
       [0.24457546],
       [0.28504607],
       [0.25106549],
       [0.84059043],
       [0.44304992],
       [0.20304673],
       [0.32580478],
       [0.28141553],
       [0.2861158 ],
       [0.49082284],
       [0.1961556 ],
       [0.17772449],
       [0.31015409],
       [0.32049753],
       [0.26352138],
       [0.23321534],
       [0.34518072],
       [0.28760908],
       [0.41806047],
       [0.19615961],
       [0.79712306],
       [0.34956156],
       [0.387056  ],
       [0.47251619],
       [0.57354785],
       [0.21298576],
       [0.38171403],
       [0.28749985],
       [0.63116246]])
#%%
#最後結果


# =============================================================================
#   #again 最後30個chemical 跑10次之後的平均結果   ei99
# =============================================================================
ei99_t_mean = np.mean(a[:, 1:], axis=1).reshape(-1,1)
#array([[0.78963499],
#       [0.920533  ],
#       [1.03606724],
#       [0.69188598],
#       [0.99555057],
#       [0.81932078],
#       [0.66660314],
#       [0.74267244],
#       [0.68193455],
#       [0.78219787],
#       [0.72839177],
#       [0.80174953],
#       [0.51142057],
#       [0.56011664],
#       [0.66988601],
#       [0.72928058],
#       [0.91557987],
#       [0.98248271],
#       [0.92406524],
#       [0.91244903],
#       [0.61223778],
#       [1.23939573],
#       [1.43233753],
#       [1.01913767],
#       [0.716442  ],
#       [1.77430842],
#       [0.5238039 ],
#       [0.68364645],
#       [0.69329105],
#       [1.18884539]])
ei99_t_std = np.std(a[:, 1:], axis=1).reshape(-1,1)
#array([[0.63257821],
#       [0.38734849],
#       [0.62273265],
#       [0.38340764],
#       [0.65202668],
#       [0.65910989],
#       [0.29337394],
#       [0.50314342],
#       [0.39903574],
#       [0.50185799],
#       [0.47307664],
#       [0.64571249],
#       [0.27824673],
#       [0.55821286],
#       [0.36677398],
#       [0.37059927],
#       [0.5028721 ],
#       [0.63063025],
#       [0.45387202],
#       [0.58364351],
#       [0.5405174 ],
#       [0.85181185],
#       [0.88949143],
#       [0.70013846],
#       [0.49100323],
#       [1.70726069],
#       [0.47007631],
#       [0.62836195],
#       [0.34000188],
#       [0.83366569]])





# =============================================================================
#   #again 最後30個chemical 跑10次之後的平均結果   recipe
# =============================================================================
recipe_t_mean = np.mean((b[:, 1:] + c[:, 1:] + d[:, 1:]), axis=1).reshape(-1,1)
#array([[0.94710683],
#       [1.13403624],
#       [0.77140737],
#       [1.11911526],
#       [1.51680098],
#       [1.09361346],
#       [0.97858373],
#       [0.97451766],
#       [1.1374171 ],
#       [0.99169592],
#       [1.16001945],
#       [1.24077291],
#       [0.92674542],
#       [0.84820459],
#       [1.10748356],
#       [0.98767945],
#       [0.97374787],
#       [1.1044324 ],
#       [1.0498703 ],
#       [1.11408474],
#       [0.95719513],
#       [1.46169089],
#       [1.21721124],
#       [1.00673303],
#       [1.17565993],
#       [1.59422204],
#       [0.76623273],
#       [0.96992666],
#       [1.06751453],
#       [1.38272382]])
recipe_t_std = np.std((b[:, 1:] + c[:, 1:] + d[:, 1:]), axis=1).reshape(-1,1)
#array([[0.29613823],
#       [0.26513372],
#       [0.31364154],
#       [0.31463023],
#       [0.81103238],
#       [0.39014891],
#       [0.29572857],
#       [0.32309539],
#       [0.33200085],
#       [0.27688379],
#       [0.46848343],
#       [0.35665564],
#       [0.22886539],
#       [0.28747664],
#       [0.3444905 ],
#       [0.3365352 ],
#       [0.28913544],
#       [0.32470181],
#       [0.29503743],
#       [0.42073142],
#       [0.27149608],
#       [0.74463141],
#       [0.51699634],
#       [0.29124166],
#       [0.5087136 ],
#       [0.8055955 ],
#       [0.17981715],
#       [0.42780423],
#       [0.37597579],
#       [0.63133913]])
#%%
    



    

    
    
    
    

similarPred_EI99T_list = list(ei99_t_mean) + [0.90985937]   # 0.90985937為回溯法做出來的結果
similarPred_RecipeT_list = list(recipe_t_mean) + [1.178416379]   # 1.178416379為回溯法做出來的結果


# visualize trifluoroacetic anhydride跟30 similar chemicals的 EI99 and ReCiPe
make_similar_prediction_into_graph(similarPred_EI99T_list, similarPred_RecipeT_list)


# 找出EI99比trifluoroacetic anhydride的chemical 以及ReCiPe
greenerEI99 = np.argwhere(ei99_t_mean < 0.90985937) + np.array([1,0])
greenerRECIPE = np.argwhere(recipe_t_mean < 1.178416379) + np.array([1,0])
# 同時EI99   & ReCiPe 都比較小的chemicals
#1 4  6  7  8  9  10 
#11 13   14   15   16    
#21   25  27  28    29

#%%
# 取得這些greener chemicals的數值
greener = ei99_t_mean[[0,3,5,6,7,8,9,10,12,13,14,15,20,24,26,27,28],[0]*17]
graph_shows_greener_substitutes_boxplot(greener)

