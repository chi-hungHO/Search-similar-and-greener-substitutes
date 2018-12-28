# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:12:16 2018

@author: e0225113
"""






#%%
# =============================================================================
# 比較pred total以及 e+ h +r 的total的表現
# =============================================================================
def compare2_total(ehr, yPredTotal, testY, title):
    

    
    mseEHR = []
    for test, pred in zip(testY, ehr):
        mseEHR += [(test-pred)**2]
    mseEHR = np.array(mseEHR)
    total_mseEHR = np.mean(mseEHR)
    print('\n------------------------- mse on EHR: ', mseEHR)
    
    
    msePred = []
    for test, pred in zip(testY, yPredTotal):
        msePred += [(test - pred)**2]
    msePred = np.array(msePred)
    total_msePred = np.mean(msePred)
    print('\n---------- mse on predicted: ', msePred)
    print('\n_____EHR mse: ', total_mseEHR)
    print('_____predicted mse: ', total_msePred, '\n\n\n')
    
    
    minn = min(min(ehr), min(yPredTotal), min(testY))
    maxx = max(max(ehr), max(yPredTotal), max(testY))
    

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    ax.plot([minn, maxx+0.2], [minn, maxx+0.2], linestyle='-', color='lightslategray', lw=1.5, label='perfect prediction line')
    ax.scatter(testY, ehr, color='green', edgecolors = 'slategray', linewidths=0.8, alpha=0.8, label='Total points (E + R +H)', s=30, marker='v')
    ax.scatter(testY, yPredTotal, color='gold', edgecolors = 'tan', linewidths=0.8 , alpha=1, label='Total points (predicted)', s = 30, marker='o')
    
    
    
#    ax.scatter(testY, ehr, color='black', edgecolors = 'slategray', linewidths=0.8, alpha=0.8, label='Total points (E + R +H)', s=30, marker='v')
#    ax.scatter(testY, yPredTotal, color='orangered', edgecolors = 'saddlebrown', linewidths=0.8 , alpha=1, label='Total points (predicted)', s = 30, marker='o')
    
    
    
    
    ax.set_xlabel('Reported value (points)', fontsize=18)
    ax.set_ylabel('Predicted value (points)', fontsize=18)

    ax.set_title(title, fontsize=18)
    
    ax.text(-0.3, 3.25, 'mse (total points E + R +H): %.3f' %(total_mseEHR), fontsize=11)
    ax.text(-0.3, 3, 'mse (total points predicted): %.3f' %(total_msePred), fontsize=11)
    
    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_ERH vs prediction')
    my_legend = ax.legend(facecolor='black', loc='upper left')

    for text in my_legend.get_texts():
        text.set_color('White')
        
    fig.savefig(title + 'ERH vs predicted total', facecolor='w')
    
    return total_mseEHR, total_msePred, mseEHR, msePred
    