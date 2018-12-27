# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:02:05 2018

@author: e0225113
"""



#%%
from __future__ import print_function
from rdkit import Chem
import pandas as pd
import os 
import numpy as np



# windows
os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\chemicals LCIA data')

lca_data = pd.read_csv('IUPAC into smiles version10.csv')




#%%
    
# 這邊無法直接用smiles變成molecule，因為有些descriptor會跑不出來
# 像是 rdMolDescriptors.CalcPBF(molecule)
# rdMolDescriptors 會遇到沒有conformation的問題
# 即便把pubchem上面給的smiles改成rdkit上面得到的smiles，也無法得到descriptors


# 解決方法
# pubchem下載的gz檔可以生成
'''https://pubchem.ncbi.nlm.nih.gov/search/#query=CC(F)F'''
# 用這個網址去loop smile後 得到他的CID
# 把全部cid變成list 用逗點隔開
# 然後再放到pubchem下載
# 再去生成descriptor

import webbrowser  

# 把ecoinvent 裡216個chemical的smiles取出來
smiles = lca_data[lca_data.columns[3]]
smiles_1 = smiles[:50]
smiles_2 = smiles[50:100]
smiles_3 = smiles[100:150]
smiles_4 = smiles[150:200]
smiles_5 = smiles[200:]


# 用來打開網頁，然後手動去複製CID 放到下一個section
for s in smiles_5:     # smiles_1  smiles_2 smiles_3    smiles_4都跑完了
    url = 'https://pubchem.ncbi.nlm.nih.gov/search/#query=' + s
    webbrowser.open_new(url)


#%%
# =============================================================================
# 從pubchem網頁上將smiles輸入後，得到的cid，存成list後，放入pubchem download下gz壓縮檔
# 再用 function_create_a_chemi_mole_to_df.py 這個檔案換成dataframe
# =============================================================================
smiles_1_cid =[6368, 1031, 8449,  7254, 8461]
smiles_1_cid += [7311, 520470,   6568,  31405,  6405]
smiles_1_cid += [6946,  8871,   31276,  7909,  70324]
smiles_1_cid += [7390,    177,  904,  176,  7918]
smiles_1_cid += [96,   6406,  180,  6367,  6326]  
smiles_1_cid += [92389, 7847,   6581,  196,  16652] 
smiles_1_cid += [31237,   7850,   7005,  7975,  24012] 
smiles_1_cid += [2124,  9989226,  24850,  222,  14013]
smiles_1_cid += [517111,  25517, 22985,  6097028,  15666]  
smiles_1_cid += [6335842,   6115,   227,  6780,  5354495]
smiles_1_cid = np.array([smiles_1_cid])

print('\n\n check duplicate  ', len(np.unique(smiles_1_cid)))

# smiles_1_cid.sdf.gz 是依照上面的smiles_1_cid這個list 裡面的CID去下載下來的檔案
smiles_1_df = create_a_chemi_mole_to_df('smiles_1_cid.sdf.gz')








smiles_2_cid = [23968,   23969,  54676860, 2256, 5462814 ]
smiles_2_cid += [10563,  6093286,  62392,  6857597,  7411  ]
smiles_2_cid += [240,  3794540, 241,  244,  7503,  ]
smiles_2_cid += [6623,   10219853,  518682,  123279,   6356 ] 
smiles_2_cid += [24408,  6358,  15920287,  7845,  7843  ]
smiles_2_cid += [ 8064 , 7844,  31272,  8846,   31288  ]
smiles_2_cid += [ 24947 , 14783,  91501,  23973,  134660]  
smiles_2_cid += [ 124202440,  10112,  24963,   8606,  280]  
smiles_2_cid += [ 6348,  281,  5943,  23706213,  24870  ]
smiles_2_cid += [ 24526,   300,  6577,  6372,  7864  ]
smiles_2_cid = np.array([smiles_2_cid])

print('\n\n check duplicate  ', len(np.unique(smiles_2_cid)))
# smiles_2_cid.sdf.gz 是依照上面的smiles_2_cid這個list 裡面的CID去下載下來的檔案
smiles_2_df = create_a_chemi_mole_to_df('smiles_2_cid.sdf.gz')







smiles_3_cid = [6945, 11734, 24638,  15910,  27375  ]
smiles_3_cid += [18541957,  517277,  23976,   311,  104730 ] 
smiles_3_cid += [14452,  14829,  24462,  23978,  16693908  ]
smiles_3_cid += [7406,   9740,  10477,  7954,  8078   ]
smiles_3_cid += [7966,  7967,  14410,  6344,  24883  ]
smiles_3_cid += [8113,  3283,  8117,  91744,   12021  ]
smiles_3_cid += [8254,  8031,  8883,  7943,  6497  ]
smiles_3_cid += [1068,  679,  31374,  6328035,  674 ] 
smiles_3_cid += [7993,  6398,  25352,  31275,  8902  ]
smiles_3_cid += [66984,   8193,  3053,  6049,  7835  ]
smiles_3_cid = np.array([smiles_3_cid])

print('\n\n check duplicate  ', len(np.unique(smiles_3_cid)))
# smiles_3_cid.sdf.gz 是依照上面的smiles_3_cid這個list 裡面的CID去下載下來的檔案
smiles_3_df = create_a_chemi_mole_to_df('smiles_3_cid.sdf.gz')





smiles_4_cid = [6324,  702,   8857,  7500,  12512  ]
smiles_4_cid += [6341,  7839,  7303,  11,  12375  ]
smiles_4_cid += [ 8071,  8076,  174,  6354,  175988]   
smiles_4_cid += [ 6325,  3301,  24524,  24617, 11137276]   
smiles_4_cid += [8607,  712,  284,  6328269,  5793]
smiles_4_cid += [753,   750,  7860,   3496,   6431 ] 
smiles_4_cid += [24842,  9321,  313,  768,  14917]
smiles_4_cid += [784,  402,  783,  785,  787  ]
smiles_4_cid += [795,   16727373,    5359967,  807,  24380]  
smiles_4_cid += [24393,  24458,   24826,  6560,  8038  ]
smiles_4_cid = np.array([smiles_4_cid])

print('\n\n check duplicate  ', len(np.unique(smiles_4_cid)))
# smiles_4_cid.sdf.gz 是依照上面的smiles_4_cid這個list 裡面的CID去下載下來的檔案
smiles_4_df = create_a_chemi_mole_to_df('smiles_4_cid.sdf.gz')





smiles_5_cid = [7892,  3776,  7915,  6363,   36679  ]
smiles_5_cid += [56841936,  612,   5352425,  11125,    433294  ]
smiles_5_cid += [224478,   23688915,  3939,  56845409,  14792,  24083  ]
smiles_5_cid = np.array([smiles_5_cid])

print('\n\n check duplicate  ', len(np.unique(smiles_5_cid)))

smiles_5_df = create_a_chemi_mole_to_df('smiles_5_cid.sdf.gz')







check_duplicate = np.concatenate((smiles_1_cid, smiles_2_cid, smiles_3_cid, smiles_4_cid, smiles_5_cid), axis=1)
print('\n\n check duplicate  ', len(np.unique(check_duplicate)))



#%%
# =============================================================================
# 將全部的dataframe合成一個之後，轉成csv然後手動將 LCIA的每個item轉成y label儲存成多個檔案
# =============================================================================

smiles_total = pd.concat([smiles_1_df, smiles_2_df, smiles_3_df, smiles_4_df, smiles_5_df], ignore_index = True)

smiles_total.to_csv('216 chemicals with descriptors.csv')






