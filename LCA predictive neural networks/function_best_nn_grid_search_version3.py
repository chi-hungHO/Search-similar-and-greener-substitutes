# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:44:31 2018

@author: e0225113
"""

 # -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:13:50 2018

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os


#%%

x = np.arange(1,101)
y = x*5

data = pd.DataFrame({'1.': y, '2.': x})

my_layer = [1,2,3]
my_neuron = [25, 50, 100, 250]

#best_nn_grid_search(data, my_layer, my_neuron)


#%%


def best_nn_grid_search_version3(allData, layer_list, neuron_list, my_yscaler):
    

    my_activation = ['relu', 'elu']
    
    allModel_R2Test_relu = {}
    allModel_MseTest_relu = {}
    
    allModel_R2Test_elu = {}
    allModel_MseTest_elu = {}

    allModel_R2Train_relu = {}
    allModel_MseTrain_relu = {}
    
    allModel_R2Train_elu = {}
    allModel_MseTrain_elu = {}
    
    
    
    

    
    
    keys_original = ['original x, original y',
'original x, standardized y',
'original x, log1p y']
    keys_std = ['standardized x, original y',
'standardized x, standardized y',
'standardized x, log1p y']
    keys_PCA99 = ['PCA (99%) x, original y',
'PCA (99%) x, standardized y',
'PCA (99%) x, log1p y']
    keys_PCA80 = ['PCA (80%) x, original y',
'PCA (80%) x, standardized y',
'PCA (80%) x, log1p y']
    
# =============================================================================
# 使用 x = original  跑neural network
# =============================================================================
    print(' now start running x = original, the input_dim of neural network is %f' % allData['standardized x, original y'][0][0].shape[1])
    input_dim =allData['standardized x, original y'][0][0].shape[1]
    for i, key in enumerate(keys_original):
        for numLayer in layer_list:
            for numNeuron in neuron_list:
                for activa in my_activation:
                    split__1 = allData[key][0]
                    split__2 = allData[key][1]
                    split__3 = allData[key][2]
                    
                    
                    my_model = Sequential()
                    if numLayer ==1:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                    elif numLayer ==2:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    elif numLayer ==3:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    else:
                        break
                    
                    my_model.add(Dense(1))
                    my_model.compile(loss ='mse', optimizer =optimizers.Adam(lr =0.01), metrics =['mse'])
                    
                    my_model1 = my_model
                    my_model2 = my_model
                    my_model3 = my_model
                    
                    my_fitModel1 = my_model1.fit(split__1[0], split__1[2], epochs = 800, verbose = 0)
                    my_fitModel2 = my_model2.fit(split__2[0], split__2[2], epochs = 800, verbose = 0)
                    my_fitModel3 = my_model3.fit(split__3[0], split__3[2], epochs = 800, verbose = 0)
                    
                    y_predict1 = my_model1.predict(split__1[1])
                    y_predict2 = my_model2.predict(split__2[1])                    
                    y_predict3 = my_model3.predict(split__3[1])
                    
                    y_predictTrain1 = my_model1.predict(split__1[0])
                    y_predictTrain2 = my_model2.predict(split__2[0])                    
                    y_predictTrain3 = my_model3.predict(split__3[0])
                    
                    if i ==1:
                        print('\n\n\nthe y should be std_y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = my_yscaler.inverse_transform(y_predict1) #當初std過的y_test變回去
                        y_inversePred2 = my_yscaler.inverse_transform(y_predict2) #當初std過的y_test變回去
                        y_inversePred3 = my_yscaler.inverse_transform(y_predict3) #當初std過的y_test變回去
                        
                        y_realTest1 = my_yscaler.inverse_transform(split__1[3]) # 當初std過的y_test變回去
                        y_realTest2 = my_yscaler.inverse_transform(split__2[3]) # 當初std過的y_test變回去
                        y_realTest3 = my_yscaler.inverse_transform(split__3[3]) # 當初std過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        y_inversePredTrain1 = my_yscaler.inverse_transform(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = my_yscaler.inverse_transform(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = my_yscaler.inverse_transform(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = my_yscaler.inverse_transform(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = my_yscaler.inverse_transform(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = my_yscaler.inverse_transform(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    
                        
                        
                        if activa == 'relu':
                            allModel_R2Test_relu[model_name] = r2Predict
                            allModel_MseTest_relu[model_name] = msePredict
                            
                            allModel_R2Train_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_elu[model_name] = r2Predict
                            allModel_MseTest_elu[model_name] = msePredict
                            
                            allModel_R2Train_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_elu[model_name] = msePredictTrain
                    

                    elif i ==2:
                        print('\n\n\nthe y should be log1p y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = np.expm1(y_predict1) #當初log1p過的y_test變回去
                        y_inversePred2 = np.expm1(y_predict2) #當初log1p過的y_test變回去
                        y_inversePred3 = np.expm1(y_predict3) #當初log1p過的y_test變回去
                        
                        y_realTest1 = np.expm1(split__1[3]) # 當初log1p過的y_test變回去
                        y_realTest2 = np.expm1(split__2[3]) # 當初log1p過的y_test變回去
                        y_realTest3 = np.expm1(split__3[3]) # 當初log1p過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        y_inversePredTrain1 = np.expm1(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = np.expm1(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = np.expm1(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = np.expm1(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = np.expm1(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = np.expm1(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    
                        
                        
                        if activa == 'relu':
                            allModel_R2Test_relu[model_name] = r2Predict
                            allModel_MseTest_relu[model_name] = msePredict
                            
                            allModel_R2Train_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_elu[model_name] = r2Predict
                            allModel_MseTest_elu[model_name] = msePredict
                            
                            allModel_R2Train_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_elu[model_name] = msePredictTrain

                    elif i ==0:
                        print('\n\n\nthe y should be original')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))

                        r2Predict_1 = r2_score(split__1[3], y_predict1)
                        r2Predict_2 = r2_score(split__2[3], y_predict2)
                        r2Predict_3 = r2_score(split__3[3], y_predict3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(split__1[3], y_predict1)
                        msePredict_2 = mean_squared_error(split__2[3], y_predict2)
                        msePredict_3 = mean_squared_error(split__3[3], y_predict3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        

                        r2PredictTrain_1 = r2_score(split__1[2], y_predictTrain1)
                        r2PredictTrain_2 = r2_score(split__2[2], y_predictTrain2)
                        r2PredictTrain_3 = r2_score(split__3[2], y_predictTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(split__1[2], y_predictTrain1)
                        msePredictTrain_2 = mean_squared_error(split__2[2], y_predictTrain2)
                        msePredictTrain_3 = mean_squared_error(split__3[2], y_predictTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    
                        
                        
                        if activa == 'relu':
                            allModel_R2Test_relu[model_name] = r2Predict
                            allModel_MseTest_relu[model_name] = msePredict
                            
                            allModel_R2Train_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_elu[model_name] = r2Predict
                            allModel_MseTest_elu[model_name] = msePredict
                            
                            allModel_R2Train_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_elu[model_name] = msePredictTrain

                    else:
                        print('\n\n\n\n\n\n\n\n\n\n\nwarning warning warning warning warning warning\n\n\n\n\n\n\n\n\n')









    allModel_R2Test_stdX_relu = {}
    allModel_MseTest_stdX_relu = {}
    
    allModel_R2Test_stdX_elu = {}
    allModel_MseTest_stdX_elu = {}

    allModel_R2Train_stdX_relu = {}
    allModel_MseTrain_stdX_relu = {}
    
    allModel_R2Train_stdX_elu = {}
    allModel_MseTrain_stdX_elu = {}






# =============================================================================
# 使用 x = standardized 跑neural network
# =============================================================================
    print(' now start running x = standardized, the input_dim of neural network is %f' % allData['standardized x, original y'][0][0].shape[1])
    input_dim =allData['standardized x, original y'][0][0].shape[1]
    for i, key in enumerate(keys_std):
        for numLayer in layer_list:
            for numNeuron in neuron_list:
                for activa in my_activation:
                    
                    split__1 = allData[key][0]
                    split__2 = allData[key][1]
                    split__3 = allData[key][2]
                    
                    
                    my_model = Sequential()
                    if numLayer ==1:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                    elif numLayer ==2:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    elif numLayer ==3:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    else:
                        break
                    
                    my_model.add(Dense(1))
                    my_model.compile(loss ='mse', optimizer =optimizers.Adam(lr =0.01), metrics =['mse'])
                    
                    
                    my_model1 = my_model
                    my_model2 = my_model
                    my_model3 = my_model
                    
                    my_fitModel1 = my_model1.fit(split__1[0], split__1[2], epochs = 800, verbose = 0)
                    my_fitModel2 = my_model2.fit(split__2[0], split__2[2], epochs = 800, verbose = 0)
                    my_fitModel3 = my_model3.fit(split__3[0], split__3[2], epochs = 800, verbose = 0)
                    
                    y_predict1 = my_model1.predict(split__1[1])
                    y_predict2 = my_model2.predict(split__2[1])                    
                    y_predict3 = my_model3.predict(split__3[1])
                    
                    y_predictTrain1 = my_model1.predict(split__1[0])
                    y_predictTrain2 = my_model2.predict(split__2[0])                    
                    y_predictTrain3 = my_model3.predict(split__3[0])
                    
                    
                    if i ==1:
                        print('\n\n\nthe y should be std_y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = my_yscaler.inverse_transform(y_predict1) #當初std過的y_test變回去
                        y_inversePred2 = my_yscaler.inverse_transform(y_predict2) #當初std過的y_test變回去
                        y_inversePred3 = my_yscaler.inverse_transform(y_predict3) #當初std過的y_test變回去
                        
                        y_realTest1 = my_yscaler.inverse_transform(split__1[3]) # 當初std過的y_test變回去
                        y_realTest2 = my_yscaler.inverse_transform(split__2[3]) # 當初std過的y_test變回去
                        y_realTest3 = my_yscaler.inverse_transform(split__3[3]) # 當初std過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        y_inversePredTrain1 = my_yscaler.inverse_transform(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = my_yscaler.inverse_transform(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = my_yscaler.inverse_transform(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = my_yscaler.inverse_transform(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = my_yscaler.inverse_transform(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = my_yscaler.inverse_transform(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    


                        
                        if activa == 'relu':
                            allModel_R2Test_stdX_relu[model_name] = r2Predict
                            allModel_MseTest_stdX_relu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_stdX_elu[model_name] = r2Predict
                            allModel_MseTest_stdX_elu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_elu[model_name] = msePredictTrain
                    

                    elif i ==2:
                        print('\n\n\nthe y should be log1p y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = np.expm1(y_predict1) #當初log1p過的y_test變回去
                        y_inversePred2 = np.expm1(y_predict2) #當初log1p過的y_test變回去
                        y_inversePred3 = np.expm1(y_predict3) #當初log1p過的y_test變回去
                        
                        y_realTest1 = np.expm1(split__1[3]) # 當初log1p過的y_test變回去
                        y_realTest2 = np.expm1(split__2[3]) # 當初log1p過的y_test變回去
                        y_realTest3 = np.expm1(split__3[3]) # 當初log1p過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        y_inversePredTrain1 = np.expm1(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = np.expm1(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = np.expm1(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = np.expm1(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = np.expm1(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = np.expm1(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        
                        if activa == 'relu':
                            allModel_R2Test_stdX_relu[model_name] = r2Predict
                            allModel_MseTest_stdX_relu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_stdX_elu[model_name] = r2Predict
                            allModel_MseTest_stdX_elu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_elu[model_name] = msePredictTrain

                    elif i ==0:
                        print('\n\n\nthe y should be original')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))

                        r2Predict_1 = r2_score(split__1[3], y_predict1)
                        r2Predict_2 = r2_score(split__2[3], y_predict2)
                        r2Predict_3 = r2_score(split__3[3], y_predict3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(split__1[3], y_predict1)
                        msePredict_2 = mean_squared_error(split__2[3], y_predict2)
                        msePredict_3 = mean_squared_error(split__3[3], y_predict3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # train data
                        print('This is for training data')
                        

                        r2PredictTrain_1 = r2_score(split__1[2], y_predictTrain1)
                        r2PredictTrain_2 = r2_score(split__2[2], y_predictTrain2)
                        r2PredictTrain_3 = r2_score(split__3[2], y_predictTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(split__1[2], y_predictTrain1)
                        msePredictTrain_2 = mean_squared_error(split__2[2], y_predictTrain2)
                        msePredictTrain_3 = mean_squared_error(split__3[2], y_predictTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        if activa == 'relu':
                            allModel_R2Test_stdX_relu[model_name] = r2Predict
                            allModel_MseTest_stdX_relu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_stdX_elu[model_name] = r2Predict
                            allModel_MseTest_stdX_elu[model_name] = msePredict
                            
                            allModel_R2Train_stdX_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_stdX_elu[model_name] = msePredictTrain

                    else:
                        print('\n\n\n\n\n\n\n\n\n\n\nwarning warning warning warning warning warning\n\n\n\n\n\n\n\n\n')        
                    
                    
                    
                    



    allModel_R2Test_PCA99_relu = {}
    allModel_MseTest_PCA99_relu = {}
    
    allModel_R2Test_PCA99_elu = {}
    allModel_MseTest_PCA99_elu = {}

    allModel_R2Train_PCA99_relu = {}
    allModel_MseTrain_PCA99_relu = {}
    
    allModel_R2Train_PCA99_elu = {}
    allModel_MseTrain_PCA99_elu = {}







# =============================================================================
# 使用 x = pca99 跑neural network
# =============================================================================
    print(' now start running x = PCA99, the input_dim of neural network is %f' % allData['PCA (99%) x, original y'][0][0].shape[1])
    input_dim =allData['PCA (99%) x, original y'][0][0].shape[1]
    for i, key in enumerate(keys_PCA99):
        for numLayer in layer_list:
            for numNeuron in neuron_list:
                for activa in my_activation:
                    
                    split__1 = allData[key][0]
                    split__2 = allData[key][1]
                    split__3 = allData[key][2]
                    
                    
                    my_model = Sequential()
                    if numLayer ==1:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                    elif numLayer ==2:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    elif numLayer ==3:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    else:
                        break
                    
                    my_model.add(Dense(1))
                    my_model.compile(loss ='mse', optimizer =optimizers.Adam(lr =0.01), metrics =['mse'])
                    
                    
                    my_model1 = my_model
                    my_model2 = my_model
                    my_model3 = my_model
                    
                    
                    my_fitModel1 = my_model1.fit(split__1[0], split__1[2], epochs = 800, verbose = 0)
                    my_fitModel2 = my_model2.fit(split__2[0], split__2[2], epochs = 800, verbose = 0)
                    my_fitModel3 = my_model3.fit(split__3[0], split__3[2], epochs = 800, verbose = 0)
                    
                    y_predict1 = my_model1.predict(split__1[1])
                    y_predict2 = my_model2.predict(split__2[1])                    
                    y_predict3 = my_model3.predict(split__3[1])


                    y_predictTrain1 = my_model1.predict(split__1[0])
                    y_predictTrain2 = my_model2.predict(split__2[0])                    
                    y_predictTrain3 = my_model3.predict(split__3[0])


                    if i ==1:
                        print('\n\n\nthe y should be std_y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = my_yscaler.inverse_transform(y_predict1) #當初std過的y_test變回去
                        y_inversePred2 = my_yscaler.inverse_transform(y_predict2) #當初std過的y_test變回去
                        y_inversePred3 = my_yscaler.inverse_transform(y_predict3) #當初std過的y_test變回去
                        
                        y_realTest1 = my_yscaler.inverse_transform(split__1[3]) # 當初std過的y_test變回去
                        y_realTest2 = my_yscaler.inverse_transform(split__2[3]) # 當初std過的y_test變回去
                        y_realTest3 = my_yscaler.inverse_transform(split__3[3]) # 當初std過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # test data
                        print('This is for training data')
                        y_inversePredTrain1 = my_yscaler.inverse_transform(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = my_yscaler.inverse_transform(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = my_yscaler.inverse_transform(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = my_yscaler.inverse_transform(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = my_yscaler.inverse_transform(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = my_yscaler.inverse_transform(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    


                        
                        
                        if activa == 'relu':
                            allModel_R2Test_PCA99_relu[model_name] = r2Predict
                            allModel_MseTest_PCA99_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA99_elu[model_name] = r2Predict
                            allModel_MseTest_PCA99_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_elu[model_name] = msePredictTrain
                    

                    elif i ==2:
                        print('\n\n\nthe y should be log1p y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = np.expm1(y_predict1) #當初log1p過的y_test變回去
                        y_inversePred2 = np.expm1(y_predict2) #當初log1p過的y_test變回去
                        y_inversePred3 = np.expm1(y_predict3) #當初log1p過的y_test變回去
                        
                        y_realTest1 = np.expm1(split__1[3]) # 當初log1p過的y_test變回去
                        y_realTest2 = np.expm1(split__2[3]) # 當初log1p過的y_test變回去
                        y_realTest3 = np.expm1(split__3[3]) # 當初log1p過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # tring data
                        print('This is for training data')
                        y_inversePredTrain1 = np.expm1(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = np.expm1(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = np.expm1(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = np.expm1(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = np.expm1(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = np.expm1(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    
                        

    
                        if activa == 'relu':
                            allModel_R2Test_PCA99_relu[model_name] = r2Predict
                            allModel_MseTest_PCA99_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA99_elu[model_name] = r2Predict
                            allModel_MseTest_PCA99_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_elu[model_name] = msePredictTrain

                    elif i ==0:
                        print('\n\n\nthe y should be original')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))

                        r2Predict_1 = r2_score(split__1[3], y_predict1)
                        r2Predict_2 = r2_score(split__2[3], y_predict2)
                        r2Predict_3 = r2_score(split__3[3], y_predict3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(split__1[3], y_predict1)
                        msePredict_2 = mean_squared_error(split__2[3], y_predict2)
                        msePredict_3 = mean_squared_error(split__3[3], y_predict3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # test data
                        print('This is for training data')
                        

                        r2PredictTrain_1 = r2_score(split__1[2], y_predictTrain1)
                        r2PredictTrain_2 = r2_score(split__2[2], y_predictTrain2)
                        r2PredictTrain_3 = r2_score(split__3[2], y_predictTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(split__1[2], y_predictTrain1)
                        msePredictTrain_2 = mean_squared_error(split__2[2], y_predictTrain2)
                        msePredictTrain_3 = mean_squared_error(split__3[2], y_predictTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        
                        if activa == 'relu':
                            allModel_R2Test_PCA99_relu[model_name] = r2Predict
                            allModel_MseTest_PCA99_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA99_elu[model_name] = r2Predict
                            allModel_MseTest_PCA99_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA99_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA99_elu[model_name] = msePredictTrain

                    else:
                        print('\n\n\n\n\n\n\n\n\n\n\nwarning warning warning warning warning warning\n\n\n\n\n\n\n\n\n')        
                    




    allModel_R2Test_PCA80_relu = {}
    allModel_MseTest_PCA80_relu = {}
    
    allModel_R2Test_PCA80_elu = {}
    allModel_MseTest_PCA80_elu = {}

    allModel_R2Train_PCA80_relu = {}
    allModel_MseTrain_PCA80_relu = {}
    
    allModel_R2Train_PCA80_elu = {}
    allModel_MseTrain_PCA80_elu = {}
    
# =============================================================================
# 使用 x = PCA80  跑neural network
# =============================================================================
    print(' now start running x = original, the input_dim of neural network is %f' % allData['PCA (80%) x, original y'][0][0].shape[1])
    input_dim =allData['PCA (80%) x, original y'][0][0].shape[1]
    for i, key in enumerate(keys_PCA80):
        for numLayer in layer_list:
            for numNeuron in neuron_list:
                for activa in my_activation:
                    
                    split__1 = allData[key][0]
                    split__2 = allData[key][1]
                    split__3 = allData[key][2]
                    
                    
                    my_model = Sequential()
                    if numLayer ==1:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                    elif numLayer ==2:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    elif numLayer ==3:
                        my_model.add(Dense(numNeuron, input_dim = input_dim, activation = activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                        my_model.add(Dense(numNeuron, activation =activa))
                    else:
                        break
                    
                    my_model.add(Dense(1))
                    my_model.compile(loss ='mse', optimizer =optimizers.Adam(lr =0.01), metrics =['mse'])
                    
                    
                    my_model1 = my_model
                    my_model2 = my_model
                    my_model3 = my_model
                    
                    
                    
                    my_fitModel1 = my_model1.fit(split__1[0], split__1[2], epochs = 800, verbose = 0)
                    my_fitModel2 = my_model2.fit(split__2[0], split__2[2], epochs = 800, verbose = 0)
                    my_fitModel3 = my_model3.fit(split__3[0], split__3[2], epochs = 800, verbose = 0)
                    
                    y_predict1 = my_model1.predict(split__1[1])
                    y_predict2 = my_model2.predict(split__2[1])                    
                    y_predict3 = my_model3.predict(split__3[1])
                    
                    y_predictTrain1 = my_model1.predict(split__1[0])
                    y_predictTrain2 = my_model2.predict(split__2[0])                    
                    y_predictTrain3 = my_model3.predict(split__3[0])

                    
                    
                    if i ==1:
                        print('\n\n\nthe y should be std_y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = my_yscaler.inverse_transform(y_predict1) #當初std過的y_test變回去
                        y_inversePred2 = my_yscaler.inverse_transform(y_predict2) #當初std過的y_test變回去
                        y_inversePred3 = my_yscaler.inverse_transform(y_predict3) #當初std過的y_test變回去
                        
                        y_realTest1 = my_yscaler.inverse_transform(split__1[3]) # 當初std過的y_test變回去
                        y_realTest2 = my_yscaler.inverse_transform(split__2[3]) # 當初std過的y_test變回去
                        y_realTest3 = my_yscaler.inverse_transform(split__3[3]) # 當初std過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # test data
                        print('This is for training data')
                        y_inversePredTrain1 = my_yscaler.inverse_transform(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = my_yscaler.inverse_transform(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = my_yscaler.inverse_transform(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = my_yscaler.inverse_transform(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = my_yscaler.inverse_transform(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = my_yscaler.inverse_transform(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        
                        if activa == 'relu':
                            allModel_R2Test_PCA80_relu[model_name] = r2Predict
                            allModel_MseTest_PCA80_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA80_elu[model_name] = r2Predict
                            allModel_MseTest_PCA80_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_elu[model_name] = msePredictTrain
                    

                    elif i ==2:
                        print('\n\n\nthe y should be log1p y')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))
                        y_inversePred1 = np.expm1(y_predict1) #當初log1p過的y_test變回去
                        y_inversePred2 = np.expm1(y_predict2) #當初log1p過的y_test變回去
                        y_inversePred3 = np.expm1(y_predict3) #當初log1p過的y_test變回去
                        
                        y_realTest1 = np.expm1(split__1[3]) # 當初log1p過的y_test變回去
                        y_realTest2 = np.expm1(split__2[3]) # 當初log1p過的y_test變回去
                        y_realTest3 = np.expm1(split__3[3]) # 當初log1p過的y_test變回去

                        r2Predict_1 = r2_score(y_realTest1, y_inversePred1)
                        r2Predict_2 = r2_score(y_realTest2, y_inversePred2)
                        r2Predict_3 = r2_score(y_realTest3, y_inversePred3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(y_realTest1, y_inversePred1)
                        msePredict_2 = mean_squared_error(y_realTest2, y_inversePred2)
                        msePredict_3 = mean_squared_error(y_realTest3, y_inversePred3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # tring data
                        print('This is for training data')
                        y_inversePredTrain1 = np.expm1(y_predictTrain1)#當初std過的y_test變回去
                        y_inversePredTrain2 = np.expm1(y_predictTrain2)#當初std過的y_test變回去
                        y_inversePredTrain3 = np.expm1(y_predictTrain3)#當初std過的y_test變回去
                        
                        y_realTrain1 = np.expm1(split__1[2]) # 當初std過的y_train變回去
                        y_realTrain2 = np.expm1(split__2[2]) # 當初std過的y_train變回去
                        y_realTrain3 = np.expm1(split__3[2]) # 當初std過的y_train變回去
                        

                        r2PredictTrain_1 = r2_score(y_realTrain1, y_inversePredTrain1)
                        r2PredictTrain_2 = r2_score(y_realTrain2, y_inversePredTrain2)
                        r2PredictTrain_3 = r2_score(y_realTrain3, y_inversePredTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(y_realTrain1, y_inversePredTrain1)
                        msePredictTrain_2 = mean_squared_error(y_realTrain2, y_inversePredTrain2)
                        msePredictTrain_3 = mean_squared_error(y_realTrain3, y_inversePredTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        
                        if activa == 'relu':
                            allModel_R2Test_PCA80_relu[model_name] = r2Predict
                            allModel_MseTest_PCA80_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA80_elu[model_name] = r2Predict
                            allModel_MseTest_PCA80_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_elu[model_name] = msePredictTrain

                    elif i ==0:
                        print('\n\n\nthe y should be original')

                    
                        print('-------------- layer: %s, Neuron: %s, activation: %s\ndata: %s' %(numLayer, numNeuron, activa, key))

                        r2Predict_1 = r2_score(split__1[3], y_predict1)
                        r2Predict_2 = r2_score(split__2[3], y_predict2)
                        r2Predict_3 = r2_score(split__3[3], y_predict3)
                        r2Predict = (r2Predict_1 + r2Predict_2 + r2Predict_3) /3
                        
                        msePredict_1 = mean_squared_error(split__1[3], y_predict1)
                        msePredict_2 = mean_squared_error(split__2[3], y_predict2)
                        msePredict_3 = mean_squared_error(split__3[3], y_predict3)
                        msePredict = (msePredict_1 + msePredict_2 + msePredict_3) /3
                        
                        print('mse are %.4f, %.4f, %.4f' %(msePredict_1, msePredict_2, msePredict_3))
                        print('\nr2 are %.4f, %.4f, %.4f' %(r2Predict_1, r2Predict_2, r2Predict_3))
                        print('---test data   R2: %s,   MSE: %s'  %(r2Predict, msePredict))
                        
                        
                        
                        
                        # test data
                        print('This is for training data')
                        

                        r2PredictTrain_1 = r2_score(split__1[2], y_predictTrain1)
                        r2PredictTrain_2 = r2_score(split__2[2], y_predictTrain2)
                        r2PredictTrain_3 = r2_score(split__3[2], y_predictTrain3)
                        r2PredictTrain = (r2PredictTrain_1 + r2PredictTrain_2 + r2PredictTrain_3) /3
                        
                        msePredictTrain_1 = mean_squared_error(split__1[2], y_predictTrain1)
                        msePredictTrain_2 = mean_squared_error(split__2[2], y_predictTrain2)
                        msePredictTrain_3 = mean_squared_error(split__3[2], y_predictTrain3)
                        msePredictTrain = (msePredictTrain_1 + msePredictTrain_2 + msePredictTrain_3) /3
                        print('---training data   R2: %s,   MSE: %s'  %(r2PredictTrain, msePredictTrain))
                        
                        
                        
                        model_name = str(numLayer) + ' Layer, ' + str(numNeuron) + ' Neurons w/ ' + '\n' + activa +', ' + key
    

                        
                        if activa == 'relu':
                            allModel_R2Test_PCA80_relu[model_name] = r2Predict
                            allModel_MseTest_PCA80_relu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_relu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_relu[model_name] = msePredictTrain
                            
                        elif activa == 'elu':
                            allModel_R2Test_PCA80_elu[model_name] = r2Predict
                            allModel_MseTest_PCA80_elu[model_name] = msePredict
                            
                            allModel_R2Train_PCA80_elu[model_name] = r2PredictTrain
                            allModel_MseTrain_PCA80_elu[model_name] = msePredictTrain

                    else:
                        print('\n\n\n\n\n\n\n\n\n\n\nwarning warning warning warning warning warning\n\n\n\n\n\n\n\n\n')  
                    
                    

                    
    return allModel_R2Test_relu,        allModel_MseTest_relu,      allModel_R2Train_relu,      allModel_MseTrain_relu,      allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,                                                                                             allModel_R2Test_stdX_relu,       allModel_MseTest_stdX_relu,         allModel_R2Train_stdX_relu,     allModel_MseTrain_stdX_relu,                                        allModel_R2Test_stdX_elu,       allModel_MseTest_stdX_elu,          allModel_R2Train_stdX_elu,      allModel_MseTrain_stdX_elu,              allModel_R2Test_PCA99_relu,      allModel_MseTest_PCA99_relu,    allModel_R2Train_PCA99_relu,     allModel_MseTrain_PCA99_relu,                                        allModel_R2Test_PCA99_elu,       allModel_MseTest_PCA99_elu,      allModel_R2Train_PCA99_elu,      allModel_MseTrain_PCA99_elu,         allModel_R2Test_PCA80_relu,      allModel_MseTest_PCA80_relu,    allModel_R2Train_PCA80_relu,     allModel_MseTrain_PCA80_relu,                                        allModel_R2Test_PCA80_elu,       allModel_MseTest_PCA80_elu,      allModel_R2Train_PCA80_elu,      allModel_MseTrain_PCA80_elu,
                
 
 
 
 
 
 

                



    
