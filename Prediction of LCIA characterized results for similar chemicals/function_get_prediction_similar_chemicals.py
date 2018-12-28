# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:49:22 2018

@author: e0225113
"""


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os





'''ei99 的部分，predicted total better than e +r + h'''
#%%

# =============================================================================
# 下面這個是用來訓練ei99 t的model，以及用來預測30 similar chemicals
# =============================================================================


def similarChemicals_ei99T(x,y, my_data):
    
#-----------------------------x
    my_xScaler = StandardScaler()
    my_xScaler = my_xScaler.fit(x)
    x_std = my_xScaler.transform(x)
    
    my_pca = PCA(n_components=0.99).fit(x_std)
    x_pca = my_pca.transform(x_std)
    
#    ---------------------------y
#    y = np.array(y).reshape(-1,1)

    y_log1p = np.log1p(y)
    
#   -----------------------------------------split
    trainX, testX, trainY, testY = train_test_split(x_pca, y_log1p, test_size=0.1, random_state=42)
    

    
#    -----------------------------------------NN model
    numFeature = x_pca.shape[1]
    
    my_model = Sequential()
    my_model.add(Dense(25, input_dim=numFeature, activation='elu'))

    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
    
#   ------------------------------------------preprocess my data
    my_dataStd = my_xScaler.transform(my_data)
    my_dataPCA = my_pca.transform(my_dataStd)
    
#    -----------------------------------------get results
    my_dataPred = my_model.predict(my_dataPCA)
    my_dataPred = np.expm1(my_dataPred)
    
    
    
    return my_model, my_dataPred









































''' recipe 的部分， e +r + h better than predicted total'''
#%%
# =============================================================================
# recipe e
#下面這個是用來訓練recipe e的model，以及用來預測30 similar chemicals
# =============================================================================
def similarChemicals_recipeE(x,y, my_data):

    
#   -----------------------------x
    my_xScaler = StandardScaler()
    my_xScaler = my_xScaler.fit(x)
    x_std = my_xScaler.transform(x)
    

    
#    ---------------------------y
#    y = np.array(y).reshape(-1,1)

    y_log1p = np.log1p(y)
    
#   -----------------------------------------split
    trainX, testX, trainY, testY = train_test_split(x_std, y_log1p, test_size=0.1, random_state=42)
    

    
#    -----------------------------------------NN model
    numFeature = x_std.shape[1]
    
    my_model = Sequential()
    my_model.add(Dense(25, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(25, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#   ------------------------------------------preprocess my data
    my_dataStd = my_xScaler.transform(my_data)

    
#    -----------------------------------------get results
    my_dataPred = my_model.predict(my_dataStd)
    my_dataPred = np.expm1(my_dataPred)
    
    return my_model, my_dataPred



#%%
# =============================================================================
# recipe h
#下面這個是用來訓練recipe h的model，以及用來預測30 similar chemicals
# =============================================================================
def similarChemicals_recipeH(x,y, my_data):

    
#   -----------------------------x
    my_xScaler = StandardScaler()
    my_xScaler = my_xScaler.fit(x)
    x_std = my_xScaler.transform(x)
    
    my_pca = PCA(n_components=0.99).fit(x_std)
    x_pca = my_pca.transform(x_std)

    
#    ---------------------------y
#    y = np.array(y).reshape(-1,1)

    y_log1p = np.log1p(y)
    
#   -----------------------------------------split
    trainX, testX, trainY, testY = train_test_split(x_pca, y_log1p, test_size=0.1, random_state=42)
    

    
#    -----------------------------------------NN model
    numFeature = x_pca.shape[1]
    
    my_model = Sequential()
    my_model.add(Dense(25, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(25, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#   ------------------------------------------preprocess my data
    my_dataStd = my_xScaler.transform(my_data)
    my_dataPCA = my_pca.transform(my_dataStd)
    
#    -----------------------------------------get results
    my_dataPred = my_model.predict(my_dataPCA)
    my_dataPred = np.expm1(my_dataPred)
    
    return my_model, my_dataPred

#%%
# =============================================================================
# recipe r
#下面這個是用來訓練recipe r的model，以及用來預測30 similar chemicals
# =============================================================================
def similarChemicals_recipeR(x,y, my_data):
    
    
#-----------------------------x
    my_xScaler = StandardScaler()
    my_xScaler = my_xScaler.fit(x)
    x_std = my_xScaler.transform(x)
    
    my_pca = PCA(n_components=0.99).fit(x_std)
    x_pca = my_pca.transform(x_std)
    
#    ---------------------------y
    my_yScaler = StandardScaler()
    y = np.array(y).reshape(-1,1)
    my_yScaler = my_yScaler.fit(y)
    y_std = my_yScaler.transform(y)
    
#   -----------------------------------------split
    trainX, testX, trainY, testY = train_test_split(x_pca, y_std, test_size=0.1, random_state=42)
    

    
#    -----------------------------------------NN model
    numFeature = x_pca.shape[1]
    
    my_model = Sequential()
    my_model.add(Dense(25, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(25, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#   ------------------------------------------preprocess my data
    my_dataStd = my_xScaler.transform(my_data)
    my_dataPCA = my_pca.transform(my_dataStd)
    
#    -----------------------------------------get results
    my_dataPred = my_model.predict(my_dataPCA)
    my_dataPred = my_yScaler.inverse_transform(my_dataPred)
    
    return my_model, my_dataPred