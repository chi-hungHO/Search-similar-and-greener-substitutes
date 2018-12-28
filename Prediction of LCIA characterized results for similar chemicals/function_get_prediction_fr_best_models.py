# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:48:50 2018

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
#%%
# =============================================================================
# 這個會有8個function，分別用來建構8個LCA prediction AI
# 每個function的input有 data
# split後
# 每個function得到的結果有 1. true training  2. predict training  3. true test   4.  predict test
# =============================================================================





# =============================================================================
# for EI99 e
# =============================================================================
def get_prediction_fr_best_ei99E(x,y):
    
    
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
    my_model.add(Dense(100, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(100, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = my_yScaler.inverse_transform(yPred)
    yPred_train =  my_yScaler.inverse_transform(yPred_train)
    
    
    trainY = my_yScaler.inverse_transform(trainY)
    testY = my_yScaler.inverse_transform(testY)
    
    return trainY, yPred_train, testY, yPred


#%%
# =============================================================================
#  for EI99 h
# =============================================================================
def get_prediction_fr_best_ei99H(x,y):

#-----------------------------x
    my_xScaler = StandardScaler()
    my_xScaler = my_xScaler.fit(x)
    x_std = my_xScaler.transform(x)
    

    
#    ---------------------------y
    y = np.array(y).reshape(-1,1)

    
#   -----------------------------------------split
    trainX, testX, trainY, testY = train_test_split(x_std, y, test_size=0.1, random_state=42)
    

    
#    -----------------------------------------NN model
    numFeature = x_std.shape[1]
    
    my_model = Sequential()
    my_model.add(Dense(50, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(50, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    return trainY, yPred_train, testY, yPred

#%%
# =============================================================================
# for EI99 R
# =============================================================================
def get_prediction_fr_best_ei99R(x,y):
    
    
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
    my_model.add(Dense(25, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = my_yScaler.inverse_transform(yPred)
    yPred_train =  my_yScaler.inverse_transform(yPred_train)
    
    
    trainY = my_yScaler.inverse_transform(trainY)
    testY = my_yScaler.inverse_transform(testY)
    
    return trainY, yPred_train, testY, yPred


#%%
# =============================================================================
# for EI99 T
# =============================================================================
def get_prediction_fr_best_ei99T(x,y):
    
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

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = np.expm1(yPred)
    yPred_train =  np.expm1(yPred_train)
    
    
    trainY = np.expm1(trainY)
    testY = np.expm1(testY)
    
    return trainY, yPred_train, testY, yPred



#%%
# =============================================================================
# for ReCiPe E
# =============================================================================
def get_prediction_fr_best_recipeE(x,y):

    
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

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = np.expm1(yPred)
    yPred_train =  np.expm1(yPred_train)
    
    
    trainY = np.expm1(trainY)
    testY = np.expm1(testY)
    
    return trainY, yPred_train, testY, yPred




#%%
# =============================================================================
# for ReCiPe H
# =============================================================================

def get_prediction_fr_best_recipeH(x,y):

    
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

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = np.expm1(yPred)
    yPred_train =  np.expm1(yPred_train)
    
    
    trainY = np.expm1(trainY)
    testY = np.expm1(testY)
    
    return trainY, yPred_train, testY, yPred

#%%
# =============================================================================
# for ReCiPe R
# =============================================================================
def get_prediction_fr_best_recipeR(x,y):
    
    
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

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = my_yScaler.inverse_transform(yPred)
    yPred_train =  my_yScaler.inverse_transform(yPred_train)
    
    
    trainY = my_yScaler.inverse_transform(trainY)
    testY = my_yScaler.inverse_transform(testY)
    
    return trainY, yPred_train, testY, yPred




#%%
# =============================================================================
# for ReCiPe T    
# =============================================================================
def get_prediction_fr_best_recipeT(x,y):
    
    
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
    my_model.add(Dense(50, input_dim=numFeature, activation='elu'))
    my_model.add(Dense(50, activation='elu'))
    my_model.add(Dense(1))
    my_model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.01), metrics=['mse'])
    my_model.fit(trainX, trainY, epochs=800, verbose=0)

    
    
#    -----------------------------------------get results
    yPred = my_model.predict(testX)
    yPred_train = my_model.predict(trainX)
    
    yPred = my_yScaler.inverse_transform(yPred)
    yPred_train =  my_yScaler.inverse_transform(yPred_train)
    
    
    trainY = my_yScaler.inverse_transform(trainY)
    testY = my_yScaler.inverse_transform(testY)
    
    return trainY, yPred_train, testY, yPred