# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 13:33:34 2018

@author: e0225113
"""

import numpy as np



#%%
# =============================================================================
# 此function是用來找data point間的距離，也就是dimension
# 這邊沒有normalization，因為只是要找相對距離，如果是要用ML預測，就得normalize，以避免某個feacture權重過大
# =============================================================================

def closest_points(data, featureStartPoint, ID_in_database, NumPoint):
    
    from math import sqrt
    import pandas as pd
    
    # 找出想要的chemical，設為target
    targetVector = data.iloc[ID_in_database].values
    
    final_distance_list = []
    
    for i in range(len(data)):  # loop 全部的chemical
        
        OtherPoinVector = data.iloc[i].values
        
        EucliDist = 0
        preEucliDist = 0
        for dim in range(featureStartPoint, len(data.columns)): # loop 每個feature or dimension
            preEucliDist +=   (targetVector[dim] - OtherPoinVector[dim])**2
        
        EucliDist = sqrt(preEucliDist)
        final_distance_list += [EucliDist]
    
    print('\n\n The shape of data here is ', data.shape)
    print('this is the 3rd one in total list: ', final_distance_list[2], '\n\n')
    
# =============================================================================
#     這邊為了方便sort，將distance的list改成 dataframe，再用sort value的功能 可以依照columns去sort 
# =============================================================================
    distDataf = pd.DataFrame(final_distance_list, columns = ['final_distance'])
    sortDistDataf = distDataf.sort_values(by=['final_distance'])
    
    #選前幾個當作similar
    similarPoint = sortDistDataf.iloc[:NumPoint]
    

    # 取得這些similar chemical的SMILES
    similar_index = similarPoint.index
    
    similarPoint = pd.DataFrame(similarPoint)
    
    
    # 因為standardize後 column會移位， standardize 後會少掉eliminate那個column，第一個column會變成SMILES
#    所以這邊用if來做格篩選，以避免出錯
    if '0,  SMILE' in data.columns:
        similar_SMILE = data.loc[similar_index, '0,  SMILE']
        similarPoint['SMILES'] = similar_SMILE
    elif '0,  SMILE' not in data.columns:
        similar_SMILE = data.iloc[similar_index, 0]
#        similar_SMILE = pd.DataFrame(similar_SMILE, columns = ['SMILES'], index = similar_index)
#        new_similarPoint = pd.concat([similarPoint, similar_SMILE], axis = 1)
        similarPoint['SMILES'] = similar_SMILE
    else:
        print('\n\n something wrong in the function "closest_points" ')


    
    print(similarPoint)
    print('\n\n The length of similarPoint is ', len(similarPoint), '\n\n\n\n')
    return similarPoint, final_distance_list
            
            
#closest_points(my_database, 9444, NumPoint)