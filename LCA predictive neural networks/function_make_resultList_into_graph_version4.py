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
def  make_resultList_into_graph_version4(result_list, title_list):
    plt.style.use('ggplot')
    
    
# =============================================================================
# relu $ original x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================
# allModel_R2Test_relu,   0     allModel_MseTest_relu,   1   allModel_R2Train_relu,  2    allModel_MseTrain_relu,   3   allModel_R2Test_elu,           allModel_MseTest_elu,       allModel_R2Train_elu,           allModel_MseTrain_elu,   7

    def relu_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_relu = {}
        ax_relu = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            #設　ｙ值以及兩個x值
            y_range = range(1, len(result_list[row,1])+1)
            test = list(result_list[row,1].values())
            training = list(result_list[row,3].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test = list(result_list[row,0].values())
            r2_train = list(result_list[row,2].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,1])+1)
            y_ticklabels = list(result_list[row,1].keys())
                
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]
            
            
            # 開始作圖
            fig_relu[row] = plt.figure(figsize = (13, 15))
            ax_relu[row] = fig_relu[row].add_subplot(111)
            # 畫一條小短線
            ax_relu[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_relu[row].scatter(test, y_range, color ='lightskyblue', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_relu[row].scatter(training, y_range, color ='crimson', alpha=0.8, label='mse of training data', s = 90, marker='v')
            
            
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_relu[row].set_yticks(y_ticks)
            ax_relu[row].set_yticklabels(new_yticklabels)
            ax_relu[row].set_xlim(-0.3, 0.5)
            ax_relu[row].set_xticks(np.linspace(0,0.5,11))
            ax_relu[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_relu[row].set_ylabel('Neural networks', fontsize=18)
            ax_relu[row].set_title(title, fontsize=22)
            
            ax_relu[row].set_xlabel('Mean squared error with ReLU & original descriptors', fontsize=18)
            ax_relu[row].set_ylim(0.3, len(result_list[row,0])+1)
            
            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > 0:
#                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_relu[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_relu[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_relu[row].legend(facecolor='black', loc ='upper left')

            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_relu

# =============================================================================
# elu & original x     這邊是用來把best_nn_grid_search這個function得到的結果elu的部分畫成lollipop
# =============================================================================
# allModel_R2Test_relu,   0     allModel_MseTest_relu,   1   allModel_R2Train_relu,  2    allModel_MseTrain_relu,   3   allModel_R2Test_elu,       4    allModel_MseTest_elu,    5   allModel_R2Train_elu,    6       allModel_MseTrain_elu,   7             
        
    def elu_graph(result_list, title_list):

        
        fig_elu = {}
        ax_elu = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            #設　ｙ值以及兩個x值
            y_range = range(1, len(result_list[row,5])+1)
            test = list(result_list[row,5].values())
            training = list(result_list[row,7].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test = list(result_list[row,4].values())
            r2_train = list(result_list[row,6].values())
            

            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1,len(result_list[row,5])+1)
            y_ticklabels = list(result_list[row,5].keys())
            
            
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' % (i +1) + label ]
                
            
            
            fig_elu[row] = plt.figure(figsize = (13, 15))
            ax_elu[row] = fig_elu[row].add_subplot(111)
            ax_elu[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            ax_elu[row].scatter(test, y_range, color ='lightskyblue', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_elu[row].scatter(training, y_range, color ='crimson', alpha=0.8, label= 'mse of training data', s = 90, marker='v')
            
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_elu[row].set_yticks(y_ticks)
            ax_elu[row].set_yticklabels(new_yticklabels)
            ax_elu[row].set_xlim(-0.3, 0.5)
            ax_elu[row].set_xticks(np.linspace(0,0.5,11))
            ax_elu[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_elu[row].set_ylabel('Neural networks', fontsize=18)

            ax_elu[row].set_title(title, fontsize=22)
            
            
#---------------------------------------------------------------------
            ax_elu[row].set_xlabel('Mean squared error with ELU & original descriptors', fontsize=18)
            ax_elu[row].set_ylim(0.3, len(result_list[row,5])+1)
#            ------------------------------------------------------------------------------------------------
            
            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > 0:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_elu[row].text(testValue +0.08, y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > 0:
                ax_elu[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))
                    

            my_legend = ax_elu[row].legend(facecolor='black', loc ='upper left')

            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_elu





# =============================================================================
# relu $ std x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



#allModel_R2Test_stdX_relu,8
#       allModel_MseTest_stdX_relu, 9
#        allModel_R2Train_stdX_relu, 10
#        allModel_MseTrain_stdX_relu, 11
#
#
#allModel_R2Test_stdX_elu, 12
#      allModel_MseTest_stdX_elu,   13
#       allModel_R2Train_stdX_elu,  14
#       allModel_MseTrain_stdX_elu, 15            



    def reluSTD_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_reluSTD = {}
        ax_reluSTD = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,9])+1)
            test = list(result_list[row,9].values())
            training = list(result_list[row,11].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test  = list(result_list[row,8].values())
            r2_train = list(result_list[row,10].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,9])+1)
            y_ticklabels = list(result_list[row,9].keys())
                
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]
#            -----------------------------------------------------------------------------------------------
            
            # 開始作圖
            fig_reluSTD[row] = plt.figure(figsize = (13, 15))
            ax_reluSTD[row] = fig_reluSTD[row].add_subplot(111)
            # 畫一條小短線
            ax_reluSTD[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_reluSTD[row].scatter(test, y_range, color ='gold', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_reluSTD[row].scatter(training, y_range, color ='green', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_reluSTD[row].set_yticks(y_ticks)
            ax_reluSTD[row].set_yticklabels(new_yticklabels)
            ax_reluSTD[row].set_xlim(-0.3, 0.5)
            ax_reluSTD[row].set_xticks(np.linspace(0,0.5,11))


            ax_reluSTD[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_reluSTD[row].set_ylabel('Neural networks', fontsize=18)

            ax_reluSTD[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_reluSTD[row].set_xlabel('Mean squared error with ReLU & standardized descriptors', fontsize=18)
            ax_reluSTD[row].set_ylim(0.3, len(result_list[row,9])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > 0:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_reluSTD[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > 0:
                ax_reluSTD[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_reluSTD[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_reluSTD





# =============================================================================
# elu $ std x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



#allModel_R2Test_stdX_relu,8
#       allModel_MseTest_stdX_relu, 9
#        allModel_R2Train_stdX_relu, 10
#        allModel_MseTrain_stdX_relu, 11
#
#
#allModel_R2Test_stdX_elu, 12
#      allModel_MseTest_stdX_elu,   13
#       allModel_R2Train_stdX_elu,  14
#       allModel_MseTrain_stdX_elu, 15     


    def eluSTD_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_eluSTD = {}
        ax_eluSTD = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,13])+1)
            test = list(result_list[row,13].values())
            training = list(result_list[row,15].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test = list(result_list[row,12].values())
            r2_train = list(result_list[row,14].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,13])+1)
            y_ticklabels = list(result_list[row,13].keys())
#            -----------------------------------------------------------------------------------------------
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]

            
            # 開始作圖
            fig_eluSTD[row] = plt.figure(figsize = (13, 15))
            ax_eluSTD[row] = fig_eluSTD[row].add_subplot(111)
            # 畫一條小短線
            ax_eluSTD[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_eluSTD[row].scatter(test, y_range, color ='gold', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_eluSTD[row].scatter(training, y_range, color ='green', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_eluSTD[row].set_yticks(y_ticks)
            ax_eluSTD[row].set_yticklabels(new_yticklabels)
            ax_eluSTD[row].set_xlim(-0.3, 0.5)
            ax_eluSTD[row].set_xticks(np.linspace(0,0.5,11))


            ax_eluSTD[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_eluSTD[row].set_ylabel('Neural networks', fontsize=18)

            ax_eluSTD[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_eluSTD[row].set_xlabel('Mean squared error with ELU & standardized descriptors', fontsize=18)
            ax_eluSTD[row].set_ylim(0.3, len(result_list[row,13])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > -0.75:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_eluSTD[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_eluSTD[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_eluSTD[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_eluSTD










# =============================================================================
# relu $ pca99 x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



# allModel_R2Test_PCA99_relu,  16
#    allModel_MseTest_PCA99_relu,   17
#    allModel_R2Train_PCA99_relu,   18
#    allModel_MseTrain_PCA99_relu,  19
#    
#    
#    
#allModel_R2Test_PCA99_elu,    20   
#    allModel_MseTest_PCA99_elu,    21
#    allModel_R2Train_PCA99_elu,    22
#    allModel_MseTrain_PCA99_elu,   23  
   


    def reluPCA99_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_reluPCA99 = {}
        ax_reluPCA99 = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,17])+1)
            test = list(result_list[row,17].values())
            training = list(result_list[row,19].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test  = list(result_list[row,16].values())
            r2_train  = list(result_list[row,18].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,17])+1)
            y_ticklabels = list(result_list[row,17].keys())
#            -----------------------------------------------------------------------------------------------
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]

            
            # 開始作圖
            fig_reluPCA99[row] = plt.figure(figsize = (13, 15))
            ax_reluPCA99[row] = fig_reluPCA99[row].add_subplot(111)
            # 畫一條小短線
            ax_reluPCA99[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_reluPCA99[row].scatter(test, y_range, color ='orangered', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_reluPCA99[row].scatter(training, y_range, color ='black', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_reluPCA99[row].set_yticks(y_ticks)
            ax_reluPCA99[row].set_yticklabels(new_yticklabels)
            ax_reluPCA99[row].set_xlim(-0.3, 0.5)
            ax_reluPCA99[row].set_xticks(np.linspace(0,0.5,11))


            ax_reluPCA99[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_reluPCA99[row].set_ylabel('Neural networks', fontsize=18)

            ax_reluPCA99[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_reluPCA99[row].set_xlabel('Mean squared error with ReLU & PCA(99%) descriptors', fontsize=18)
            ax_reluPCA99[row].set_ylim(0.3, len(result_list[row,17])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > -0.75:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_reluPCA99[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_reluPCA99[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_reluPCA99[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_reluPCA99






# =============================================================================
# elu $ pca99 x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



# allModel_R2Test_PCA99_relu,  16
#    allModel_MseTest_PCA99_relu,   17
#    allModel_R2Train_PCA99_relu,   18
#    allModel_MseTrain_PCA99_relu,  19
#    
#    
#    
#allModel_R2Test_PCA99_elu,    20   
#    allModel_MseTest_PCA99_elu,    21
#    allModel_R2Train_PCA99_elu,    22
#    allModel_MseTrain_PCA99_elu,   23  
   


    def eluPCA99_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_eluPCA99 = {}
        ax_eluPCA99 = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,21])+1)
            test = list(result_list[row,21].values())
            training = list(result_list[row,23].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test  = list(result_list[row,20].values())
            r2_train  = list(result_list[row,22].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,21])+1)
            y_ticklabels = list(result_list[row,21].keys())
#            -----------------------------------------------------------------------------------------------
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]

            
            # 開始作圖
            fig_eluPCA99[row] = plt.figure(figsize = (13, 15))
            ax_eluPCA99[row] = fig_eluPCA99[row].add_subplot(111)
            # 畫一條小短線
            ax_eluPCA99[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_eluPCA99[row].scatter(test, y_range, color ='orangered', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_eluPCA99[row].scatter(training, y_range, color ='black', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_eluPCA99[row].set_yticks(y_ticks)
            ax_eluPCA99[row].set_yticklabels(new_yticklabels)
            ax_eluPCA99[row].set_xlim(-0.3, 0.5)
            ax_eluPCA99[row].set_xticks(np.linspace(0,0.5,11))

            ax_eluPCA99[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_eluPCA99[row].set_ylabel('Neural networks', fontsize=18)

            ax_eluPCA99[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_eluPCA99[row].set_xlabel('Mean squared error with ELU & PCA(99%) descriptors', fontsize=18)
            ax_eluPCA99[row].set_ylim(0.3, len(result_list[row,21])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > -0.75:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_eluPCA99[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_eluPCA99[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_eluPCA99[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_eluPCA99




# =============================================================================
# relu $ pca80 x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



# allModel_R2Test_PCA80_relu,  24
#    allModel_MseTest_PCA80_relu,    25
#    allModel_R2Train_PCA80_relu,    26
#    allModel_MseTrain_PCA80_relu,   27                                     
#    
#    
#    
# allModel_R2Test_PCA80_elu,   28
#    allModel_MseTest_PCA80_elu,      29
#    allModel_R2Train_PCA80_elu,      30
#    allModel_MseTrain_PCA80_elu      31
   


    def reluPCA80_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_reluPCA80 = {}
        ax_reluPCA80 = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,25])+1)
            test = list(result_list[row,25].values())
            training = list(result_list[row,27].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test  = list(result_list[row,24].values())
            r2_train  = list(result_list[row,26].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,25])+1)
            y_ticklabels = list(result_list[row,25].keys())
#            -----------------------------------------------------------------------------------------------
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]

            
            # 開始作圖
            fig_reluPCA80[row] = plt.figure(figsize = (13, 15))
            ax_reluPCA80[row] = fig_reluPCA80[row].add_subplot(111)
            # 畫一條小短線
            ax_reluPCA80[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_reluPCA80[row].scatter(test, y_range, color ='turquoise', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_reluPCA80[row].scatter(training, y_range, color ='blue', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_reluPCA80[row].set_yticks(y_ticks)
            ax_reluPCA80[row].set_yticklabels(new_yticklabels)
            ax_reluPCA80[row].set_xlim(-0.3, 0.5)
            ax_reluPCA80[row].set_xticks(np.linspace(0,0.5,11))


            ax_reluPCA80[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_reluPCA80[row].set_ylabel('Neural networks', fontsize=18)

            ax_reluPCA80[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_reluPCA80[row].set_xlabel('Mean squared error with ReLU & PCA(80%) descriptors', fontsize=18)
            ax_reluPCA80[row].set_ylim(0.3, len(result_list[row,25])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > -0.75:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_reluPCA80[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_reluPCA80[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_reluPCA80[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_reluPCA80











# =============================================================================
# relu $ pca80 x    這邊是用來把best_nn_grid_search這個function得到的結果relu的部分畫成lollipop
# =============================================================================



# allModel_R2Test_PCA80_relu,  24
#    allModel_MseTest_PCA80_relu,    25
#    allModel_R2Train_PCA80_relu,    26
#    allModel_MseTrain_PCA80_relu,   27                                     
#    
#    
#    
# allModel_R2Test_PCA80_elu,   28
#    allModel_MseTest_PCA80_elu,      29
#    allModel_R2Train_PCA80_elu,      30
#    allModel_MseTrain_PCA80_elu      31
   


    def eluPCA80_graph(result_list, title_list):
        
        # 先創兩個空個字典，用來放圖片
        fig_eluPCA80 = {}
        ax_eluPCA80 = {}
        for row, title in zip(range(1, len(result_list)), title_list):
            
            
        
            #設　ｙ值以及兩個x值
#            -----------------------------------------------------------------------------------------
            y_range = range(1, len(result_list[row,29])+1)
            test = list(result_list[row,29].values())
            training = list(result_list[row,31].values())
            
            # 這兩個值是用來當作text放在點圖旁邊的
            r2_test  = list(result_list[row,28].values())
            r2_train  = list(result_list[row,30].values())
            
            # 設定y label，在原本的neural network名字之前加上數字，方便辨識
            y_ticks = np.arange(1, len(result_list[row,29])+1)
            y_ticklabels = list(result_list[row,29].keys())
#            -----------------------------------------------------------------------------------------------
            
            new_yticklabels = []
            for i, label in enumerate(y_ticklabels):
                new_yticklabels += ['(%s).   ' %(i +1) + label ]

            
            # 開始作圖
            fig_eluPCA80[row] = plt.figure(figsize = (13, 15))
            ax_eluPCA80[row] = fig_eluPCA80[row].add_subplot(111)
            # 畫一條小短線
            ax_eluPCA80[row].hlines( y =y_range, xmin =test, xmax =training, color = 'grey', alpha=0.8)
            # 製造小圓點
            ax_eluPCA80[row].scatter(test, y_range, color ='turquoise', alpha=0.8, label='mse of test data', s = 90, marker = "o")
            ax_eluPCA80[row].scatter(training, y_range, color ='blue', alpha=0.8, label='mse of training data', s = 90, marker='v')
            # 這邊要記得先把y tick設成想要的數量後，才可以貼上名字
            ax_eluPCA80[row].set_yticks(y_ticks)
            ax_eluPCA80[row].set_yticklabels(new_yticklabels)
            ax_eluPCA80[row].set_xlim(-0.3, 0.5)
            ax_eluPCA80[row].set_xticks(np.linspace(0,0.5,11))


            ax_eluPCA80[row].tick_params(axis = 'y', labelsize = 10)
            
            ax_eluPCA80[row].set_ylabel('Neural networks', fontsize=18)

            ax_eluPCA80[row].set_title(title, fontsize=22)
            
            
            
            
#---------------------------------------------------------------------
            ax_eluPCA80[row].set_xlabel('Mean squared error with ELU & PCA(80%) descriptors', fontsize=18)
            ax_eluPCA80[row].set_ylim(0.3, len(result_list[row,29])+1)
#            ------------------------------------------------------------------------------------------------
            

            # 在bar上面加上legend  y_pos就是y 的位置，也將是 np.arange(1,12)
            for y_pos, (testValue, trainValue, mseTestValue, mseTrainValue) in enumerate(zip(test, training, r2_test, r2_train)):
#                if testValue > -0.75:
                    # 在value往左0.0005的地方，以及y_post的地方加上label
                ax_eluPCA80[row].text(testValue + 0.08 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'test R$^2$: %.2f' %mseTestValue))
#                if trainValue > -0.75:
                ax_eluPCA80[row].text(trainValue -0.2 , y_pos +1.3, str('(%s)   ' %(y_pos +1) + r'training R$^2$: %.2f' %mseTrainValue))

            my_legend = ax_eluPCA80[row].legend(facecolor='black', loc ='upper left')



            for text in my_legend.get_texts():
                text.set_color("White")
                
            plt.tight_layout()
    
        return fig_eluPCA80









    
    fig_relu = relu_graph(result_list, title_list)
    fig_elu = elu_graph(result_list, title_list)    
    
    
    fig_reluSTD = reluSTD_graph(result_list, title_list)  
    fig_eluSTD = eluSTD_graph(result_list, title_list)  
    
    
    fig_reluPCA99 = reluPCA99_graph(result_list, title_list)  
    fig_eluPCA99 = eluPCA99_graph(result_list, title_list)  
    
    
    fig_reluPCA80 = reluPCA80_graph(result_list, title_list)  
    fig_eluPCA80 = eluPCA80_graph(result_list, title_list)  
    
    
    
    
    

    
    
    os.chdir('C:\\Users\\NUS_2\\Desktop\\Chi-Hung 2\\借我跑_____\\results')
    
    for i in range(1, len(fig_relu)+1):
        fig_relu[i].savefig(title_list[i-1] + '_relu_' +str(i))
    for i in range(1, len(fig_elu)+1):
        fig_elu[i].savefig(title_list[i-1] + '_elu_' +str(i))
        
        
    for i in range(1, len(fig_reluSTD)+1):
        fig_reluSTD[i].savefig(title_list[i-1] + '_reluSTD_' +str(i))
    for i in range(1, len(fig_eluSTD)+1):
        fig_eluSTD[i].savefig(title_list[i-1] + '_eluSTD_' +str(i))
        
        
    for i in range(1, len(fig_reluPCA99)+1):
        fig_reluPCA99[i].savefig(title_list[i-1] + '_reluPCA99_' +str(i))
    for i in range(1, len(fig_eluPCA99)+1):
        fig_eluPCA99[i].savefig(title_list[i-1] + '_eluPCA99_' +str(i))
        
    
    for i in range(1, len(fig_reluPCA80)+1):
        fig_reluPCA80[i].savefig(title_list[i-1] + '_reluPCA80_' +str(i))
    for i in range(1, len(fig_eluPCA80)+1):
        fig_eluPCA80[i].savefig(title_list[i-1] + '_eluPCA80_' +str(i))
    
    
    
    return fig_relu, fig_elu,                  fig_reluSTD, fig_eluSTD,                fig_reluPCA99,  fig_eluPCA99,                               fig_reluPCA80, fig_eluPCA80
    
    
    
#%%
    
