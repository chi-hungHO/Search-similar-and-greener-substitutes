# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:48:33 2018

@author: e0225113
"""



#%%

def get_data_treatedPCA(data, dim):
    
    from sklearn.decomposition import PCA
    
    pca_model = PCA(n_components = dim).fit(data)
    new_data = pca_model.transform(data)
    

#    　因為ｓｃｉｐｙ的版本有變，所以’ｍｌｅ’這個選項不能用了　　要另外加參數
    pca_model2 = PCA(n_components = 'mle', svd_solver='full').fit(data)
    new_data2 = pca_model2.transform(data)
    

    
    pca_model3 = PCA(n_components = min(data.shape[0], data.shape[1])).fit(data)
    new_data3 = pca_model3.transform(data)
    
    print('\n my method pca: \n', new_data.shape, '\n mle: \n', new_data2.shape, '\n min(n_samples, n_features): \n', new_data3.shape)    
    return new_data
    