# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:41:06 2018

@author: e0225113
"""






import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os 



#%%


def best_ensemble_grid_search(x,y):
    
    # 創造空的字典，用來放最後結果
    predict_results ={}
    r2Test_results ={}
    mseTest_results ={}
    r2Train_results ={}
    mseTrain_results ={}
    
# 原始data，1model
# x 分成有PCA以及std， y分成original，std 以及 log1p，6個model
# 共7 model

# 然後再分成用 lgb跟xgb 所以總共用 2*(1+6) = 14個model
    
    #將x分成original   std  以及 pca
    x_original = x
    
    x_scaler = StandardScaler()
    x_scaler = x_scaler.fit(x)
    x_std = x_scaler.transform(x)
    
    x_pca = get_data_treatedPCA(x_std, 0.99)
    
    #將y分成original   std  以及 log1p
    y_original = np.array([y]).reshape(-1,1)
    y_scaler = StandardScaler()
    y_scaler = y_scaler.fit(y_original)
    y_std = y_scaler.transform(y_original)
    
    y_log1p = np.log1p(y_original)
    
    plt.figure(1)
    sns.distplot(y_log1p, fit=norm, kde=False)
    plt.ylabel('EI99 total points with log1p')
    plt.xlabel('216 training data for LCA predictive neural networks')
    
    plt.figure(2)
    sns.distplot(y_std, fit=norm, kde=False)
    plt.ylabel('EI99 total points with standardization')
    plt.xlabel('216 training data for LCA predictive neural networks')
    
    plt.figure(3)
    sns.distplot(y_original, fit=norm, kde=False)
    plt.ylabel('EI99 total points (original)')
    plt.xlabel('216 training data for LCA predictive neural networks')
    
    
# =============================================================================
# scenario 1 x=original  y=original
# =============================================================================
    xOrigina_train, xOriginal_test, yOriginal_train, yOroginal_test =train_test_split(x_original, y_original, test_size=0.2, random_state=42)
# =============================================================================
# scenario 2 x=std  y=original
# =============================================================================
    xStd_train, xStd_test, yOriginal_train, yOriginal_test =train_test_split(x_std, y_original, test_size=0.2, random_state=42)
# =============================================================================
# scenario 3 x=std  y=std
# =============================================================================
    xStd_train, xStd_test, yStd_train, yStd_test =train_test_split(x_std, y_std, test_size=0.2, random_state=42)
# =============================================================================
# scenario 4  x=std  y=log1p
# =============================================================================
    xStd_train, xStd_test, yLog1p_train, yLog1p_test =train_test_split(x_std, y_log1p, test_size=0.2, random_state=42)
# =============================================================================
# scenario 5  x=pca   y=original
# =============================================================================
    xPca_train, xPca_test, yOriginal_train, yOriginal_test =train_test_split(x_pca, y_original, test_size=0.2, random_state=42)
# =============================================================================
# scenario 6  x=pca  y=std
# =============================================================================
    xPca_train, xPca_test, yStd_train, yStd_test =train_test_split(x_pca, y_std, test_size=0.2, random_state=42)
# =============================================================================
# scenario 7  x=pca  y=log1p
# =============================================================================
    xPca_train, xPca_test, yLog1p_train, yLog1p_test =train_test_split(x_pca, y_log1p, test_size=0.2, random_state=42)


    # 將上面的train test data放到list裡面方便loop，這邊 xStd_train有重複，其實兩者是一樣的，雖然是在不同的scenario下面split，
    # 但是因為test_size以及random_state固定，所以分到的data會一樣
    xTrain = [xOrigina_train,   xStd_train,         xStd_train, xStd_train,     xPca_train,         xPca_train, xPca_train]
    xTest = [xOriginal_test,    xStd_test,          xStd_test,  xStd_test,      xPca_test,          xPca_test,  xPca_test]
    yTrain = [yOriginal_train,  yOriginal_train,    yStd_train, yLog1p_train,   yOriginal_train,    yStd_train, yLog1p_train]
    yTest = [yOroginal_test,    yOriginal_test,     yStd_test,  yLog1p_test,    yOriginal_test,     yStd_test,  yLog1p_test]
    
    
    
    

    
    
# =============================================================================
# 分成兩組去跑  1.xgb當作model
# =============================================================================
    name_xgb = ['x: original, y: original, xgboost']
    for xName in ['x: standardized, ', 'x: PCA, ']:
        for yName in ['y: original, xgboost', 'y: standardized, xgboost', 'y: log1p, xgboost']:
            name_xgb += [xName + yName]
            

    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    
    
    for i, (name, a, b, c, d) in enumerate(zip(name_xgb, xTrain, xTest, yTrain, yTest)):
        if i == 2 or i == 5:  # i==2  5的時候是yTrain = std的時候，目的是為了辨識yTrain =std，方便之後復原的原來的樣子
            print('\n\n i =' ,i)
            print('check y_std')
            
            model_xgb.fit(a,c)
            pred = y_scaler.inverse_transform(model_xgb.predict(b)) # 先預測後，再用inverse_transform復原成原來的樣子
            
            predict_results[name] = pred #把結果放進去字典裡
            
            real_y = y_scaler.inverse_transform(d) # 當初std過的y_test變回去
            
            r2 = r2_score(real_y, pred)
            mse = mean_squared_error(real_y, pred)
            r2Test_results[name] = r2  #把結果放進去字典裡
            mseTest_results[name] = mse   #把結果放進去字典裡
            
            
            
            
            
            
            # 把training data再放進去model一次，主要目的是比較training data的r2  mse
            pred_train = y_scaler.inverse_transform(model_xgb.predict(a))
            real_train_y = y_scaler.inverse_transform(c)
            
            r2_train = r2_score(real_train_y, pred_train)
            mse_train = mean_squared_error(real_train_y, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
        
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)
        elif i == 3 or i == 6:  # i==3  6的時候是yTrain = log1p的時候，目的是為了辨識yTrain =log1p，方便之後復原的原來的樣子
            print('\n\n i =' ,i)
            print('check y_log1p')
            
            model_xgb.fit(a,c)
            pred = np.expm1(model_xgb.predict(b))  # 先預測後，再用expm1復原成原來的樣子
            
            predict_results[name] = pred#把結果放進去字典裡
            
            real_y = np.expm1((d))  # 當初log1p過的y_test變回去
            
            r2 = r2_score(real_y, pred)
            mse = mean_squared_error(real_y, pred)
            r2Test_results[name] = r2  #把結果放進去字典裡
            mseTest_results[name] = mse   #把結果放進去字典裡
            
            
            
            # 把training data再放進去model一次，主要目的是比較training data的r2  mse
            pred_train = np.expm1(model_xgb.predict(a))
            real_train_y = np.expm1(c)
            
            r2_train = r2_score(real_train_y, pred_train)
            mse_train = mean_squared_error(real_train_y, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)
        else:   # i==0 1 4的時候是yTrain = original的時候，目的是為了辨識yTrain =original，方便之後復原的原來的樣子
            print('\n\n i =' ,i)
            print('check y_original')
            
            model_xgb.fit(a,c)
            pred = model_xgb.predict(b)
            
            predict_results[name] = pred
            
            
            r2 = r2_score(d, pred)
            mse = mean_squared_error(d, pred)
            r2Test_results[name] = r2
            mseTest_results[name] = mse

            
     
            
            pred_train = model_xgb.predict(a)
            
            r2_train = r2_score(c, pred_train)
            mse_train = mean_squared_error(c, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)




# =============================================================================
# lgb當作model
# =============================================================================
            
    # 因為lgb要求y label的部分，data的形式只能是1D np array 或者pd series 或者list，所以在特地把y的部分reshape
    y_original_new = np.array([y]).reshape(216,)
    y_scaler = StandardScaler()
    y_scaler = y_scaler.fit(y_original)
    y_std = y_scaler.transform(y_original).reshape(216,)
    
    y_log1p = np.log1p(y_original).reshape(216,)
    
    
    #也因此split的部分要重設
# =============================================================================
# scenario 1
# =============================================================================
    xOrigina_train, xOriginal_test, yOriginal_train, yOroginal_test =train_test_split(x_original, y_original_new, test_size=0.2, random_state=42)
# =============================================================================
# scenario 2
# =============================================================================
    xStd_train, xStd_test, yOriginal_train, yOriginal_test =train_test_split(x_std, y_original_new, test_size=0.2, random_state=42)
# =============================================================================
# scenario 3
# =============================================================================
    xStd_train, xStd_test, yStd_train, yStd_test =train_test_split(x_std, y_std, test_size=0.2, random_state=42)
# =============================================================================
# scenario 4
# =============================================================================
    xStd_train, xStd_test, yLog1p_train, yLog1p_test =train_test_split(x_std, y_log1p, test_size=0.2, random_state=42)
# =============================================================================
# scenario 5
# =============================================================================
    xPca_train, xPca_test, yOriginal_train, yOriginal_test =train_test_split(x_pca, y_original_new, test_size=0.2, random_state=42)
# =============================================================================
# scenario 6
# =============================================================================
    xPca_train, xPca_test, yStd_train, yStd_test =train_test_split(x_pca, y_std, test_size=0.2, random_state=42)
# =============================================================================
# scenario 7
# =============================================================================
    xPca_train, xPca_test, yLog1p_train, yLog1p_test =train_test_split(x_pca, y_log1p, test_size=0.2, random_state=42)
    
    
    # 將上面的train test data放到list裡面方便loop，這邊 xStd_train有重複，其實兩者是一樣的，雖然是在不同的scenario下面split，
    # 但是因為test_size以及random_state固定，所以分到的data會一樣
    xTrain = [xOrigina_train,   xStd_train,         xStd_train, xStd_train,     xPca_train,         xPca_train, xPca_train]
    xTest = [xOriginal_test,    xStd_test,          xStd_test,  xStd_test,      xPca_test,          xPca_test,  xPca_test]
    yTrain = [yOriginal_train,  yOriginal_train,    yStd_train, yLog1p_train,   yOriginal_train,    yStd_train, yLog1p_train]
    yTest = [yOroginal_test,    yOriginal_test,     yStd_test,  yLog1p_test,    yOriginal_test,     yStd_test,  yLog1p_test]
    
        
        
    name_lgb = ['x: original, y: original, lightgbm']
    for xName in ['x: standardized, ', 'x: PCA, ']:
        for yName in ['y: original, lightgbm', 'y: standardized, lightgbm', 'y: log1p, lightgbm']:
            name_lgb += [xName + yName]
            
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                          learning_rate=0.05, n_estimators=720,
                          max_bin = 55, bagging_fraction = 0.8,
                          bagging_freq = 5, feature_fraction = 0.2319,
                          feature_fraction_seed=9, bagging_seed=9,
                          min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

        
    for i, (name, a, b, c, d) in enumerate(zip(name_lgb, xTrain, xTest, yTrain, yTest)):
        if i == 2 or i == 5: # i==2  5的時候是yTrain = std的時候，目的是為了辨識yTrain =std，方便之後復原的原來的樣子
            print('\n\n i =' ,i)
            print('check y_std')
            
            model_lgb.fit(a,c)
            pred = y_scaler.inverse_transform(model_lgb.predict(b))   # 先預測後，再用inverse_transform復原成原來的樣子
            
            predict_results[name] = pred
            
            real_y = y_scaler.inverse_transform(d)  # 把原本的y_test  inverse_transform回去
            
            r2 = r2_score(real_y, pred)
            mse = mean_squared_error(real_y, pred)
            r2Test_results[name] = r2
            mseTest_results[name] = mse

            
            
            
            pred_train = y_scaler.inverse_transform(model_lgb.predict(a))
            real_train_y = y_scaler.inverse_transform(c)
            
            r2_train = r2_score(real_train_y, pred_train)
            mse_train = mean_squared_error(real_train_y, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
        
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)
        elif i == 3 or i == 6:   # i==3  6的時候是yTrain = log1p的時候，目的是為了辨識yTrain =log1p，方便之後復原的原來的樣子
            print('\n\n i =' ,i)
            print('check y_log1p')
            
            model_lgb.fit(a,c)
            pred = np.expm1(model_lgb.predict(b))   # 先預測後，再用expm1復原成原來的樣子
            
            predict_results[name] = pred
            
            real_y = np.expm1((d))
            
            r2 = r2_score(real_y, pred)
            mse = mean_squared_error(real_y, pred)
            r2Test_results[name] = r2
            mseTest_results[name] = mse
            
            
            
            
            pred_train = np.expm1(model_lgb.predict(a))
            real_train_y = np.expm1(c)
            
            r2_train = r2_score(real_train_y, pred_train)
            mse_train = mean_squared_error(real_train_y, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)
        else:
            print('\n\n i =' ,i)
            print('check y_original')
            
            model_lgb.fit(a,c)
            pred = model_lgb.predict(b)
            
            predict_results[name] = pred
            
            
            r2 = r2_score(d, pred)
            mse = mean_squared_error(d, pred)
            r2Test_results[name] = r2
            mseTest_results[name] = mse


            
            
            pred_train = model_lgb.predict(a)
            
            r2_train = r2_score(c, pred_train)
            mse_train = mean_squared_error(c, pred_train)
            r2Train_results[name] = r2_train
            mseTrain_results[name] = mse_train
            
            print(name, ', R2:  ', r2, ', MSE:  ', mse)
            print('train', ', R2:  ', r2_train, ', MSE:  ', mse_train)
        
        
        
        
        
    # 分成5個結果，後面是個是主要的，前面那一個是用來得到預測後的結果，準備用來畫點的散佈圖的
    return predict_results, r2Test_results, mseTest_results, r2Train_results, mseTrain_results
        
        
        






    
    
    
    


