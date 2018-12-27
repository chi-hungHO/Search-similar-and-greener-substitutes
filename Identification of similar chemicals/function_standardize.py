# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:29:59 2018

@author: e0225113
"""


import pandas as pd
import numpy as np

#%%
def standardize_my_data(data):
    
    from scipy.stats import zscore
    
    
# =============================================================================
#     # eliminate non-numeric   np.number 表示datatype，不可以用float 不然會失去一些columns
# =============================================================================
    numericCol  = data.select_dtypes(include=[np.number]).columns
    data_numeric = data[numericCol]
    
    # 將標準差為0的columns去除，才可以做zscore
    for i in data_numeric.columns:
        if data_numeric[i].std() ==0:
            data_numeric = data_numeric.drop([i], axis = 1)
        else:
            continue
            
    
# =============================================================================
# 用來找zcore的
# =============================================================================
    data_standardized = data_numeric.apply(zscore)  #得到normalize後的
    
    
# =============================================================================
#     #現在要把SMILE那個column加回去
# =============================================================================
    stringCol = data.select_dtypes(include = object).columns
    if len(stringCol) > 0:
        data_string = data[stringCol]
        data_standardized = pd.concat([data_string, data_standardized], axis =1 )
    
    return data_standardized