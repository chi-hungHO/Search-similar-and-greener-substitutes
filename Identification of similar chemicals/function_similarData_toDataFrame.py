# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:16:17 2018

@author: e0225113
"""


import os 
import numpy as np
import pandas as pd

#windows
os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\pubchem full data\\full data in sdf')

#%%



def similarData_toDataFrame(data, InexTargetChemical):
    
    def get_similarData(data):
    
        #將第一行空格刪掉
        data = data.drop(['Unnamed: 0'], axis = 1)
        
        if len(data) > 10**6:
        # 將會變成過大的數字轉成一個numpy可以接受，但數值依然很大的範圍
            data.iloc[24757,6] = 1.1166964083149396e+150   # 這樣值還是很大，不過在範圍內了
        
        # 標準化
        data_std = standardize_my_data(data)
        
        # 準備PCA，所以先把非int or np.number的cell去掉，例如string
        print('shape of standardized data: ' , data_std.shape)
        data_prePCA = data_std.select_dtypes(include = [np.number])
        print('shape of standardized data after removing the column which include strings: ', data_prePCA.shape)
        # 用來找PCA後需要多少的dimension比較適合
        find_pca_dimension(data_prePCA)
        # 得到PCA後的data，不過會失去columns
        data_pca = get_data_treatedPCA(data_prePCA, 80) # 這個data_pca只是一個numpy矩陣
        print('after pca, the shape of data is ', data_pca.shape, ' (excluding SMILEs)')
        
    
        # 將data_pca從矩陣轉成 dataframe，再補上SMILEs
        data_pca = pd.DataFrame(data_pca)
        smiles = data[data.columns[1]]
        data_pca = pd.concat([smiles, data_pca], axis = 1)
        print('to see if Trifluoroacetic anhydride is at the correct position: ', data_pca.iloc[InexTargetChemical, 0]) #確認這個位置是不是　Trifluoroacetic anhydride
        
    #    # 比較原始dataset，standardize dataset以及pca後的dataset所得到的距離
    #    similarPoints, totalDistance = closest_points(data, featureStartPoint=2, ID_in_database=InexTargetChemical, NumPoint=30)
    #    # standardize後
    #    std_similarPoints, std_totalDistance = closest_points(data_std, featureStartPoint=0, ID_in_database=InexTargetChemical, NumPoint=30)
        # PCA後
        pca_similarPoints, pca_totalDistance = closest_points(data_pca, featureStartPoint=1, ID_in_database=InexTargetChemical, NumPoint=30)
        
        indexPca_similarPoints = pca_similarPoints.index
        
        similarData = data.loc[indexPca_similarPoints]
        
        return similarData
    
    def clean_similarData(similarData_original):
        
        similarData_clean = similarData_original.drop(['to_be_eliminated', '0,  SMILE'], axis =1)
        return similarData_clean
    
    
    similarData_original = get_similarData(data)
    similarData_clean = clean_similarData(similarData_original)
    
    print('Hey dude now we have 30 data points similar to our target, the shape of data is ""%s""' %str(similarData_clean.shape) )
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_30_similarPoints')
    
    similarData_original.to_csv('chemical %s similarPoint_original.csv' %InexTargetChemical)
    similarData_clean.to_csv('chemical %s similarPoint_clean.csv' %InexTargetChemical)
    
    return similarData_original, similarData_clean
        
        
    
    
    
    