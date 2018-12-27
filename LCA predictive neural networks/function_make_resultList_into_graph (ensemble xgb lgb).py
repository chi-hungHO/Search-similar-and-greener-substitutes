# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:34:57 2018

@author: e0225113
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os





#%%
def make_resultList_into_graph_ensemble(result_list, title_list):
    plt.style.use('ggplot')
    
    
# =============================================================================
#     這邊是用來把best_ensemble_grid_search這個function得到的結果畫成lollipop
# =============================================================================
    
    # 先創兩個空個字典，用來放圖片
    fig_ensemble = {}
    ax_ensemble = {}
    # 下面這個row代表不同組data，像我的data有8組， row 就是從1-8   題外話len(result_list) =9
    for row, title in zip(range(1, len(result_list)), title_list):
        print('check   test mse: ', result_list[row,2], '\n\n\n\n\n train mse: ', result_list[row,4], '\n\n\n\n\n')
        
        #設　ｙ值以及兩個x值
        y_range = range(1, len(result_list[row,1])+1)  # len(result_list[row,1] =14，所以後面要再+1
        test = list(result_list[row,2].values())  # 2是test  4是training   mse
        training = list(result_list[row,4].values())
        
        # 這兩個mse值是用來當作text放在點圖旁邊的
        r2_test = list(result_list[row,1].values())  # 1是test   3是training  r2
        r2_train = list(result_list[row,3].values())
        
        # 設定y label，在原本的neural network名字之前加上數字，方便辨識
        y_ticks = np.arange(1,len(result_list[row,1])+1)
        y_ticklabels = list(result_list[row,1].keys())
        
        # 重新設定y ticklabel，讓y ticklabel不會那麼長，影響圖的寬度
        y_ticklabels = ['x: original, y: original\n xgboost',
         'x: standardized, y: original\n xgboost',
         'x: standardized, y: standardized\n xgboost',
         'x: standardized, y: log1p\n xgboost',
         'x: PCA, y: original\n xgboost',
         'x: PCA, y: standardized\n xgboost',
         'x: PCA, y: log1p\n xgboost',
         'x: original, y: original\n lightgbm',
         'x: standardized, y: original\n lightgbm',
         'x: standardized, y: standardized\n lightgbm',
         'x: standardized, y: log1p\n lightgbm',
         'x: PCA, y: original\n lightgbm',
         'x: PCA, y: standardized\n lightgbm',
         'x: PCA, y: log1p\n lightgbm']
        
        
        # 在每個 y ticklabel上面加上編號，方便辨識
        new_yticklabels = []
        for i, label in enumerate(y_ticklabels):
            new_yticklabels += ['(%s).   ' %(i +1) + label ]
            
        
        # 開始作圖
        fig_ensemble[row] = plt.figure(figsize = (13, 6))
        ax_ensemble[row] = fig_ensemble[row].add_subplot(111)
        # 畫一條小短線
        ax_ensemble[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
        # 製造小圓點
        ax_ensemble[row].scatter(test, y_range, color ='pink', alpha=0.8, label='mse of test data', s = 90, marker = "o")
        ax_ensemble[row].scatter(training, y_range, color ='purple', alpha=0.8, label= 'mse of training data', s = 90, marker='v')
        
        
        # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
        ax_ensemble[row].set_yticks(y_ticks)
        ax_ensemble[row].set_yticklabels(new_yticklabels)
        ax_ensemble[row].set_xlim(-2, 15)
        ax_ensemble[row].set_ylim(0.5, len(result_list[row,0])+1)
        ax_ensemble[row].set_xticks([int(0), 0.46, 1.92, 3.63, 6.38, 10.45,15])
        ax_ensemble[row].set_xticklabels([0, 0.46, 1.92, 3.63, 6.38, 10.45,15])
        ax_ensemble[row].tick_params(axis = 'y', labelsize = 10)
        
        ax_ensemble[row].set_ylabel('Models of ensemble learning', fontsize=18)
        ax_ensemble[row].set_xlabel('Mean squared error', fontsize=18)
        ax_ensemble[row].set_title(title, fontsize=22)
        
        
        for y_pos, (testValue, trainValue, r2TestValue, r2TrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
            ax_ensemble[row].text(testValue + 2, y_pos + 1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %r2TestValue))
            ax_ensemble[row].text(trainValue -1.5 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %r2TrainValue))
        
        
        
#        # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
#        for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):  # 其實這個mseTrainValue根本沒用到顆顆
#            if testValue > 0:
#                # 在value往左0.0005的地方，以及y_post的地方加上label
                

        my_legend = ax_ensemble[row].legend(facecolor='black', loc ='upper left')

        for text in my_legend.get_texts():
            text.set_color("White")
            
        plt.tight_layout()
        
        
    os.chdir('G:\\My Drive\\NUS\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\NN_noKFold_results\\ensemble results')
    
    for i in range(1, len(fig_ensemble)+1):
        fig_ensemble[i].savefig('ensemble_' +str(i))
        


    return fig_ensemble
    

