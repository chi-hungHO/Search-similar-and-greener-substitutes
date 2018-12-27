#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:48:03 2018

@author: chiouthebiya
"""

import matplotlib.pyplot as plt

#%%
def graph_show_best_model_new(mseList, modelList, title):
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    
    colorList = ['gold', 'gold', 'green', 'green', 'orangered', 'orangered', 'dimgrey', 'dimgrey', 'lightskyblue']
    legendColor = ['gold','green','orangered','dimgrey', 'lightskyblue']

    
    x = range(1,10)
    ax.bar(x, mseList, color=colorList, alpha=0.8)
    
    ax.set_xticks(range(1,10))
    ax.set_xticklabels(modelList, rotation=70)
    ax.set_xlabel('Best model for each of the categories', fontsize=16)
    ax.set_ylabel('Mean squared error' ,fontsize =16)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(0,0.5)
    ax.set_yticks(np.linspace(0.05,0.5, 10))
    
    
#    for windows
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_best_models')
    
#    for mac 
#    os.chdir('/Volumes/GoogleDrive/我的雲端硬碟/NUS/Paper in Wang\'s lab/papers/2018 11 4 LCA_AI 2/Results_best_models')
    
    my_proxy = []
    for i in legendColor:
        my_proxy += [mlines.Line2D([], [], color='none', marker='o', markerfacecolor=i)]

    my_legend = ax.legend(my_proxy,['Original features','Standardized\nfeatures','PCA(99%) features','PCA(80%) features','ensemble learning'], loc=1, facecolor='black')
    
    for text in my_legend.get_texts():
        text.set_color("White")
    
    
    plt.tight_layout()
    fig.savefig('Best model %s' %title, facecolor='white')