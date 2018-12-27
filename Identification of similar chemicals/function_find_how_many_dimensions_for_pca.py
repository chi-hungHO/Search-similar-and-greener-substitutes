# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:16:23 2018

@author: e0225113
"""


#%%
# =============================================================================
# 用來找PCA後需要多少的dimension比較適合
# =============================================================================

def find_pca_dimension(data):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('seaborn')

    
    pca_model = PCA().fit(data)
    explained_variance_ratio = pca_model.explained_variance_ratio_
    
    fig1 = plt.figure(1, figsize = (10,5))
    ax1 = fig1.add_subplot(111)
    ax1.bar([i  for i in range(len(data.columns))], explained_variance_ratio,  color = 'crimson', alpha=0.8)
    ax1.plot([i  for i in range(len(data.columns))], np.cumsum(explained_variance_ratio), color = 'dodgerblue',alpha=0.8, linewidth=5, linestyle='--')
    ax1.set_xlabel('Number of components', fontsize=18)
    ax1.set_xticks([1]+ [i for i in range(10,130,10)] + [125])
    ax1.set_ylabel('Cumulative explained variance', fontsize =18)
    ax1.set_title('Selection of number of principle components', fontsize=22)
    
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_PCA_how_many_dimenstions')
    fig1.savefig('number_pca_dimensions', facelor='white')
    
    plt.tight_layout()