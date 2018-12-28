# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:56:48 2018

@author: e0225113
"""



import matplotlib.pyplot as plt
import matplotlib as mpl







#%%
    
def graph_shows_greener_substitutes_boxplot(data):
    fig = plt.figure(figsize=(15,3))
    ax = fig.add_subplot(111)
    
    x = np.linspace(0,2.58,1000)
    color1 = [plt.cm.YlGnBu(i) for i in np.linspace(0,1,1000)]
    for i,cc in zip(x,color1):
        my_bar = ax.plot([i,i], [0,2], color =cc)
    
    
    
    boxprops = dict(linewidth=3, color='dimgrey')
    
    
    ax.boxplot(data, vert=False, showfliers=False, whis = 'range', widths =1.6, boxprops=boxprops, medianprops = boxprops, whiskerprops = boxprops, capprops = boxprops)
    
    
    ax.set_xlim(0,2.583)
    ax.set_ylim(0,2)
    
    x_pos = [1.45e-2, 1.68e-1, 5.02e-1, 6.44e-1, 8.54e-1, 1.2, 1.58, 1.73, 2.58]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['0.0145\nButane', '0.1680\nDichloromethane', '0.5020\nBenzaldehyde', '0.6440\nGlycine', '0.8540\nPhenyl isocyanate', '1.20\nDiethyl ether', '1.58\nTrifluoroacetic acid', '1.73\nHexafluoroethane', '2.58\nPerfluoropentane'], fontsize=14)
    ax.set_yticks([])
    ax.tick_params(axis='x', rotation=60)
    
    ax.set_title('Low impact                                                                                                                                                                                                                                                                               High impact', fontsize=10)

    
    
    for i in x_pos:
        ax.plot([i,i], [0,2], color ='silver')
    
# =============================================================================
# 使用proxy製作legend，精簡版本，
# 先創造一個空的list，然後創造物件mlines.Line2D([], [], color='none', marker='o', markerfacecolor=i)
# 記得要先把物件放到list []裡面，才可以放到一開始的空的proxy list裡面
# =============================================================================
    my_proxy = []
    my_proxy += [mlines.Line2D([], [], color='none', marker='s', markerfacecolor='dimgrey')]
    my_legend = fig.legend(my_proxy,['Chemicals greener than the target chemical'], loc = 'upper center')
    


    
    plt.tight_layout()
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_30_similarPoints_LCIA')
    fig.savefig('green substitutes in boxplot', facecolr='white')



