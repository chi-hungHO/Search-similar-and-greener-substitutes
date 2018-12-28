# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:23:28 2018

@author: e0225113
"""


#%%
# =============================================================================
# 要return mse, mape, scatter圖  以及 每個點的mse mape
# =============================================================================
def get_mape_mse_n_scatter_selectedRange(x, y, trainY, yPred_train, testY, yPred, title):
    

    
    n = len(testY)
    
    mse = []
    for test, pred in zip(testY, yPred):
        mse += [(test-pred)**2]
    mse = np.array(mse)
    total_mse = np.mean(mse)
    print('\n-------------- mse on test data: ', mse)
    
    
    mse_train = []
    for ytrain, ytrain_predin in zip(trainY, yPred_train):
        mse_train += [(ytrain - ytrain_predin)**2]
    total_mse_train = np.mean(mse_train)
    
    
    
    mape =[]
    for test, pred in zip(testY, yPred):
        mape += [np.abs((test-pred) / test)]
    mape = np.array(mape)
    total_mape = mape.sum()*100/n
    print('\n_____ total mse: ', total_mse)
    print('\n_____ total mape: ', total_mape)
    

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    ax.plot([y.min(), y.max()+0.2], [y.min(), y.max()+0.2], linestyle='-', color='lightslategray', lw=1.5, label='perfect prediction line')
    ax.scatter(trainY, yPred_train, color='black', edgecolors = 'slategray', linewidths=0.8, alpha=0.8, label='training chemicals', s=30, marker='v')
    ax.scatter(testY, yPred, color='orangered', edgecolors = 'saddlebrown', linewidths=0.8 , alpha=1, label='testing chemicals', s=30, marker='o')
    
    ax.set_xlabel('Reported value (points)', fontsize=18)
    ax.set_ylabel('Predicted value (points)', fontsize=18)


#-------------------------------------------
    ax.set_title(title, fontsize=22)
    ax.set_xlim(-0.05,1)
    ax.set_ylim(-0.05,1)

    ax.text(0, 0.78, 'mse on test data: %.3f' %(total_mse), fontsize=11)
    ax.text(0, 0.74, 'mse on training data: %.3f' %(total_mse_train), fontsize=11)
#    --------------------------------------------------
    
    
    
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_best_scatter\\smaller range')
    my_legend = ax.legend(facecolor='black', loc='upper left')

    for text in my_legend.get_texts():
        text.set_color('White')
        
    fig.savefig(title + 'scatter (selected range)', facecolor='w')
    
    return total_mse, total_mape, mse, mape
    