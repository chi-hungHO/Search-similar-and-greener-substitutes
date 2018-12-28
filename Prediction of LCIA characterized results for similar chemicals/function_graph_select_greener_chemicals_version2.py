# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:52:39 2018

@author: e0225113
"""


#%%
def make_similar_prediction_into_graph(ei99, recipe):
    
    x = range(1,32)

    
    

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    
    for xx, y1, y2 in zip(x, ei99, recipe):
        ax.plot([xx,xx], [y1,y2], color = 'grey', alpha=0.5, linewidth=5)

    ax.fill_between(range(1,34), 0.90985937, 0, color='tomato', alpha = 0.3, label='\nRange lower than EI99 total of trifluoroacetic anhydride')
    ax.fill_between(range(-2,32), 1.178416379, 0, color='lightsteelblue', alpha = 0.4,  label='\nRange lower than ReCiPe total of trifluoroacetic anhydride')

    
        
    ax.scatter(x, ei99, s =50, color='orangered', alpha=1, marker="o", label='\nEI99 total of similar chemicals')
    ax.scatter(x, recipe, s =50, color='black', alpha=0.8, marker="o", label='\nReCiPe total of similar chemicals')
    
    ax.scatter(31, 0.90985937, s =150, color='darkorange', alpha=1, marker="s", label='\nEI99 total of trifluoroacetic anhydride')
    ax.scatter(31, 1.178416379, s =150, color='steelblue', alpha=1, marker="s", label='\nReCiPe total of trifluoroacetic anhydride')
    
    ax.set_xlim(0,32)
    ax.set_ylim(0,2.5)
    ax.set_xticks(range(1,32))
    ax.set_xticklabels(list(range(1,31)) + ['trifluoroacetic\nanhydride'], rotation=50, fontsize=12)
#    ax.set_yticklabels([0,1,2,3,4,5], fontsize=12)
    
    ax.set_xlabel('30 chemicals most similar to trifluoroacetic anhydride', fontsize=18)
    ax.set_ylabel("EI99 / ReCiPe total points", fontsize=18)
    ax.set_title('Exploration for greener substitutes', fontsize=22)
    
    my_legend = ax.legend(facecolor='white', loc='upper left')
    
#    for text in my_legend.get_texts():
#        text.set_color('white')

    os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\Results_30_similarPoints_LCIA')
    

    plt.tight_layout()
    fig.savefig('green substitute new version2', facecolr='white')
    

##%%
#similarPred_RecipeT_list = list(similarPred_RecipeE + similarPred_RecipeH + similarPred_RecipeR) + [1.178416379]
#similarPred_EI99T_list = list(similarPred_EI99T) + [0.90985937]
#
#
#
#make_similar_prediction_into_graph(similarPred_EI99T_list, similarPred_RecipeT_list)