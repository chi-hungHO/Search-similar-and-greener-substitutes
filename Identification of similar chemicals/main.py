# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:50:07 2018

@author: e0225113
"""

#%%
# =============================================================================
# 要跑這個 需要用rdkit 的environment!!!
# =============================================================================
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors



import os
import numpy as np
import pandas as pd

import gzip
import pickle



os.chdir('G:\\My Drive\\NUS\\Paper in Wang\'s lab\\papers\\2018 11 4 LCA_AI 2\\pubchem full data\\full data in sdf')
os.getcwd()

#%%
# =============================================================================
# 以下只是用來做前置作業
# =============================================================================
# =============================================================================
# Below is just for preprocessing.
# =============================================================================
# =============================================================================
# 這部分是使用 function_create_a_chemi_mole_to_df 這個檔案將gz變成dataframe，同時產生molecular descriptors
# =============================================================================
file_name_1 = '1-50.sdf.gz'
file_name_2 = '1-200.sdf.gz'
file_name_3 = '1-25000.sdf.gz'
file_name_4 = '1-100000.sdf.gz'
file_name_5 = '1-500000.sdf.gz'
file_name_6 = '500001-1000000.sdf.gz'
file_name_7 = '1000001-1500000.sdf.gz'
file_name_8 = '1500001-2000000.sdf.gz'
file_name_9 = '2000001-2500000.sdf.gz'
file_name_10 = '2500001-3000000.sdf.gz'


fileName = file_name_5
my_database = create_a_chemi_mole_to_df(fileName)



# =============================================================================
# 這部分是使用 function_convert_gz_to_df 這個檔案將gz變成dataframe之後，再將dataframe變成CSV 這樣之後就可以直接import csv不用跑分子特性的輸出了
# =============================================================================
# 用來分割名稱，方便csv取名
a = fileName.split('.sdf')
a = a[0]
my_database.to_csv( a +'.csv')




# =============================================================================
# 純粹為了loop 儲存每一個csv
# =============================================================================

file_name_1 = '1-50.sdf.gz'
file_name_2 = '1-200.sdf.gz'
file_name_3 = '1-25000.sdf.gz'
file_name_4 = '1-100000.sdf.gz'

lst = [file_name_1, file_name_2, file_name_3, file_name_4]
for i in lst:
    my_database = create_a_chemi_mole_to_df(i)
    a = i.split('.sdf')
    a = a[0]
    my_database.to_csv( a +'.csv')

# =============================================================================
# 將CSV導入之後，將它全部疊在一起
# =============================================================================
new_my_database_csv_1 = pd.read_csv('1-500000.csv')
new_my_database_csv_2 = pd.read_csv('500001-1000000.csv')
new_my_database_csv_3 = pd.read_csv('1000001-1500000.csv')
new_my_database_csv_4 = pd.read_csv('1500001-2000000.csv')

# ingore_index = True 是用來讓index自動往下排
new_my_database_csv_2million = new_my_database_csv_1.append(new_my_database_csv_2, ignore_index = True)
new_my_database_csv_2million = new_my_database_csv_2million.append(new_my_database_csv_3, ignore_index = True)
new_my_database_csv_2million = new_my_database_csv_2million.append(new_my_database_csv_4, ignore_index = True)



new_my_database_csv_2million = new_my_database_csv_2million.drop(['Unnamed: 0'], axis = 1)

# =============================================================================
# 以上只是前置作業，正式從下面開始
# =============================================================================
# =============================================================================
# The above is just for preprocessing, please start from below
# =============================================================================


 






#%%
# step 2-8

# =============================================================================
# 這部分是將data從CSV import進來，不需要經過之前的跑分子的特性模擬

# 這邊不儲存standardize後的data，一律使用原始csv import進來，再去standardize
# =============================================================================
# 我們的target 
# Trifluoroacetic anhydride pubchemID 9845 csv 9843 這邊pubchem ID有問題，所以有出入
# DMSO pubchemID 679   csv 678
# Chloropyrazine pubchemID 73277 csv  73257     # C1=CN=C(C=N1)Cl



new_my_database_csv = pd.read_csv('1-2000000.csv')

new_my_database_csv = new_my_database_csv.drop(['Unnamed: 0'], axis = 1)




#%%
# =============================================================================
# 做PCA 然後比較 原始距離，std距離 以及pca距離
# =============================================================================
# 這邊會遇到一個問題 "RuntimeWarning:overflow encountered in double_scalars."
# 是因為在chemical  index 24757 這個地方 在column 6 Ipc 值非常大，因為在算distance的時候，squre以後會超過numpy能給的範圍(-1.79769313486e+308, 1.79769313486e+308). 所以會出現Inf   表示infinity   只要把這個值改成其他數字就可以解決這個bug

# 不過即便不改也不會怎麼樣
# 但是我們還是改吧
# 將會變成過大的數字轉成一個numpy可以接受，但數值依然很大的範圍
new_my_database_csv.iloc[24757,6] = 1.1166964083149396e+150   # 這樣值還是很大，不過在範圍內了

# 標準化
data_standardized = standardize_my_data(new_my_database_csv)




# PCA
# 準備PCA，所以先把非int or np.number的cell去掉，例如string
print('shape of original data_standardized: ' , data_standardized.shape)
data_standardized = data_standardized.select_dtypes(include = [np.number])
print('shape of data_standardized after removing the column which include strings: ', data_standardized.shape)
# 用來找PCA後需要多少的dimension比較適合
find_pca_dimension(data_standardized)
# 得到PCA後的data，不過會失去columns
data_pca = get_data_treatedPCA(data_standardized, 80) # 這個data_pca只是一個numpy矩陣
# =============================================================================
# 將data_pca從矩陣轉成 dataframe，再補上SMILEs
# =============================================================================
data_pca = pd.DataFrame(data_pca)
smiles = new_my_database_csv[new_my_database_csv.columns[1]]
data_pca = pd.concat([smiles, data_pca], axis = 1)
print('to see if Trifluoroacetic anhydride is at the correct position: ', data_pca.iloc[9843, 0]) #確認這個位置是不是　Trifluoroacetic anhydride





# 比較原始dataset，standardize dataset以及pca後的dataset所得到的距離
similarPoints, totalDistance = closest_points(new_my_database_csv, featureStartPoint=2, ID_in_database=9843, NumPoint=30)
# standardize後
stand_similarPoints, stand_totalDistance = closest_points(data_standardized, featureStartPoint=0, ID_in_database=9843, NumPoint=30)
# 這邊也會出現一個error，就是上面那個inf造成的，把那個改掉之後就沒事了，不過不改也不會怎麼樣
# PCA後
pca_similarPoints, pca_totalDistance = closest_points(data_pca, featureStartPoint=1, ID_in_database=9843, NumPoint=30)


# 兩百萬筆資料 standardize前的similarity結果
# 使用的data為 new_my_database_csv，也就是'1-2000000.csv'

#         final_distance                                 SMILES
#9843           0.000000            O=C(OC(=O)C(F)(F)F)C(F)(F)F
#67865         22.973595            O=C(NC(=O)C(F)(F)F)C(F)(F)F
#73686         29.266362            O=C(CC(=O)C(F)(F)F)C(F)(F)F
#84621         46.031818            O=C(C=C(O)C(F)(F)F)C(F)(F)F
#586103        49.661277            NC(=CC(=O)C(F)(F)F)C(F)(F)F
#67772         58.161463            CC(=CC(=O)C(F)(F)F)C(F)(F)F
#82650         71.228770             FC(F)=C(F)OC(F)(F)C(F)(F)F
#12265         74.833835               O=NN(CC(F)(F)F)CC(F)(F)F
#549748        78.373876            O=C(C=C(S)C(F)(F)F)C(F)(F)F
#518301        82.308016    O[C-]([CH-][C-](O)C(F)(F)F)C(F)(F)F
#593078        84.356567           O=C(C=C(Cl)C(F)(F)F)C(F)(F)F
#75178         90.313041  O=C([O-])C(F)(F)C(F)(F)C(F)(F)F.[Na+]
#136200        93.339923                 COC(=O)CNC(=O)C(F)(F)F
#81843         95.150855                 C=CC(=O)OCC(F)(F)C(F)F
#145122       104.176554               CC(S)=CC(=O)C(F)(F)C(F)F
#80344        105.876069            O=S(=O)(OCC(F)(F)F)C(F)(F)F
#100849       113.033091              C=CCOC(=O)C(F)(F)C(F)(F)F
#525750       113.715595               CCCOC(=O)C(F)(F)C(F)(F)F
#67773        114.156052                 CCOC(=O)CC(=O)C(F)(F)F
#251463       116.450888                 CN(N=O)C(=O)NCC(F)(F)F
#120210       116.767926          O=C(F)C(F)(F)C(F)(F)OC(F)(F)F
#92464        118.884497            FC(F)=C(F)OC(F)(F)C(F)(F)Cl
#1588771      119.332724              CC(C)NC(=O)NC(=O)C(F)(F)F
#67724        120.258326              C=CC(=O)OCC(F)(F)C(F)(F)F
#68018        120.271411              CC(C)CC(=O)CC(=O)C(F)(F)F
#1588770      120.580040                 CCNC(=O)NC(=O)C(F)(F)F
#136181       120.583618                 CCOC(=O)C=C(N)C(F)(F)F
#533179       122.806193            CCN(CC)C(=O)NS(F)(F)(F)(F)F
#67868        122.821676                O=C(OCC(F)(F)F)C(F)(F)F
#136182       123.067299                  CCOC(=O)CC(O)C(F)(F)F


# 兩百萬筆資料 standardize後的similarity結果
# 使用的data為 new_my_database_csv，也就是'1-2000000.csv'

#         final_distance                            SMILES
#9843           0.000000       O=C(OC(=O)C(F)(F)F)C(F)(F)F
#67868          2.543550           O=C(OCC(F)(F)F)C(F)(F)F
#69614          2.849491      O=C(NOC(=O)C(F)(F)F)C(F)(F)F
#100849         2.980866         C=CCOC(=O)C(F)(F)C(F)(F)F
#520048         3.067655   C=COC(=O)C(F)(F)C(F)(F)C(F)(F)F
#188429         3.108446       C=C(F)C(=O)OC(F)(F)C(C)(F)F
#67908          3.141293           CCOC(=O)C(F)(F)C(F)(F)F
#137018         3.174561     CCOC(=O)C=C(C(F)(F)F)C(F)(F)F
#67721          3.197827     COC(=O)C(F)(F)C(F)(F)C(F)(F)F
#67724          3.202335         C=CC(=O)OCC(F)(F)C(F)(F)F
#1751786        3.318036      O=C([O-])C(F)(F)C(F)(F)C(F)F
#140676         3.323168          COC(=O)C(F)(F)C(F)=C(F)F
#9781           3.329794            COC(=O)C(F)(F)C(F)(F)F
#75076          3.343646      C=CC(=O)OC(C(F)(F)F)C(F)(F)F
#123488         3.366185      C=C(C)C(=O)OCC(F)(F)C(F)(F)F
#525750         3.397314          CCCOC(=O)C(F)(F)C(F)(F)F
#9642           3.477574    CCOC(=O)C(F)(F)C(F)(F)C(F)(F)F
#533605         3.499352  C=CCOC(=O)C(F)(F)C(F)(F)C(F)(F)F
#9775           3.523315      O=C(O)C(F)(F)C(F)(F)C(F)(F)F
#67903          3.604207  C=CC(=O)OCC(F)(F)C(F)(F)C(F)(F)F
#145376         3.645022      COC(=O)CC(=O)C(F)(F)C(F)(F)F
#1751787        3.651344         O=C(O)C(F)(F)C(F)(F)C(F)F
#84621          3.658453       O=C(C=C(O)C(F)(F)F)C(F)(F)F
#549314         3.723787       O=C(Cl)ON(C(F)(F)F)C(F)(F)F
#73686          3.745363       O=C(CC(=O)C(F)(F)F)C(F)(F)F
#76449          3.760712   C=C(C)C(=O)OC(C(F)(F)F)C(F)(F)F
#87125          3.769214        C=CC(=O)OCCC(F)(F)C(F)(F)F
#549069         3.790323  C=C(C)C(=O)OCC(F)(F)C(F)C(F)(F)F
#549722         3.806794    O=C(OCCOC(=O)C(F)(F)F)C(F)(F)F
#67772          3.828972       CC(=CC(=O)C(F)(F)F)C(F)(F)F




# 兩百萬筆資料 standardize後，並且PCA後的similarity結果
# 使用的data為 new_my_database_csv，也就是'1-2000000.csv'

#         final_distance                            SMILES
#9843           0.000000       O=C(OC(=O)C(F)(F)F)C(F)(F)F
#67868          2.534066           O=C(OCC(F)(F)F)C(F)(F)F
#69614          2.847363      O=C(NOC(=O)C(F)(F)F)C(F)(F)F
#100849         2.948805         C=CCOC(=O)C(F)(F)C(F)(F)F
#188429         3.062312       C=C(F)C(=O)OC(F)(F)C(C)(F)F
#520048         3.063204   C=COC(=O)C(F)(F)C(F)(F)C(F)(F)F
#67908          3.121124           CCOC(=O)C(F)(F)C(F)(F)F
#137018         3.155037     CCOC(=O)C=C(C(F)(F)F)C(F)(F)F
#67724          3.167023         C=CC(=O)OCC(F)(F)C(F)(F)F
#67721          3.188533     COC(=O)C(F)(F)C(F)(F)C(F)(F)F
#140676         3.291533          COC(=O)C(F)(F)C(F)=C(F)F
#1751786        3.303248      O=C([O-])C(F)(F)C(F)(F)C(F)F
#9781           3.323085            COC(=O)C(F)(F)C(F)(F)F
#75076          3.325137      C=CC(=O)OC(C(F)(F)F)C(F)(F)F
#123488         3.328042      C=C(C)C(=O)OCC(F)(F)C(F)(F)F
#525750         3.370495          CCCOC(=O)C(F)(F)C(F)(F)F
#9642           3.472611    CCOC(=O)C(F)(F)C(F)(F)C(F)(F)F
#533605         3.498465  C=CCOC(=O)C(F)(F)C(F)(F)C(F)(F)F
#9775           3.500162      O=C(O)C(F)(F)C(F)(F)C(F)(F)F
#67903          3.598805  C=CC(=O)OCC(F)(F)C(F)(F)C(F)(F)F
#1751787        3.626600         O=C(O)C(F)(F)C(F)(F)C(F)F
#84621          3.631940       O=C(C=C(O)C(F)(F)F)C(F)(F)F
#145376         3.633544      COC(=O)CC(=O)C(F)(F)C(F)(F)F
#549314         3.693387       O=C(Cl)ON(C(F)(F)F)C(F)(F)F
#87125          3.729000        C=CC(=O)OCCC(F)(F)C(F)(F)F
#73686          3.734849       O=C(CC(=O)C(F)(F)F)C(F)(F)F
#76449          3.744405   C=C(C)C(=O)OC(C(F)(F)F)C(F)(F)F
#549069         3.770147  C=C(C)C(=O)OCC(F)(F)C(F)C(F)(F)F
#67772          3.805929       CC(=CC(=O)C(F)(F)F)C(F)(F)F
#549722         3.809457    O=C(OCCOC(=O)C(F)(F)F)C(F)(F)F









# 這邊會發現，2million data跟 0.5million data在絕對距離上不會有差別，但是在std過後的相對距離上會有差別
# 下面為 0.5 million資料的結果





# 50萬筆資料 standardize前的similarity結果
# 使用的資料:new_my_database_csv_1，也就是'1-500000.csv'

#        final_distance                                 SMILES
#9843          0.000000            O=C(OC(=O)C(F)(F)F)C(F)(F)F
#67865        22.973595            O=C(NC(=O)C(F)(F)F)C(F)(F)F
#73686        29.266362            O=C(CC(=O)C(F)(F)F)C(F)(F)F
#84621        46.031818            O=C(C=C(O)C(F)(F)F)C(F)(F)F
#67772        58.161463            CC(=CC(=O)C(F)(F)F)C(F)(F)F
#82650        71.228770             FC(F)=C(F)OC(F)(F)C(F)(F)F
#12265        74.833835               O=NN(CC(F)(F)F)CC(F)(F)F
#75178        90.313041  O=C([O-])C(F)(F)C(F)(F)C(F)(F)F.[Na+]
#136200       93.339923                 COC(=O)CNC(=O)C(F)(F)F
#81843        95.150855                 C=CC(=O)OCC(F)(F)C(F)F
#145122      104.176554               CC(S)=CC(=O)C(F)(F)C(F)F
#80344       105.876069            O=S(=O)(OCC(F)(F)F)C(F)(F)F
#100849      113.033091              C=CCOC(=O)C(F)(F)C(F)(F)F
#67773       114.156052                 CCOC(=O)CC(=O)C(F)(F)F
#251463      116.450888                 CN(N=O)C(=O)NCC(F)(F)F
#120210      116.767926          O=C(F)C(F)(F)C(F)(F)OC(F)(F)F
#92464       118.884497            FC(F)=C(F)OC(F)(F)C(F)(F)Cl
#67724       120.258326              C=CC(=O)OCC(F)(F)C(F)(F)F
#68018       120.271411              CC(C)CC(=O)CC(=O)C(F)(F)F
#136181      120.583618                 CCOC(=O)C=C(N)C(F)(F)F
#67868       122.821676                O=C(OCC(F)(F)F)C(F)(F)F
#136182      123.067299                  CCOC(=O)CC(O)C(F)(F)F
#120201      125.459000          O=C(O)C(F)(F)C(F)(F)OC(F)(F)F
#100947      125.868144               C=CCOC(F)(F)C(F)C(F)(F)F
#342408      126.675132                 CCOC(=O)C=C(C)C(F)(F)F
#238353      126.890598               CCCCC(=O)C(F)(F)C(F)(F)F
#443289      127.360754               O=C(O)CCC(=O)C(Cl)C(=O)O
#257190      127.454610                N=C(OCC(F)(F)F)C(F)(F)F
#239950      127.612994                 CC(C)S(=O)CCC(N)C(=O)O
#238241      127.894101             CC(C)=CC(=O)CC(=O)C(F)(F)F

# 50萬筆資料 standardize後的similarity結果
# 使用的資料:new_my_database_csv_1，也就是'1-500000.csv'


#        final_distance                               SMILES
#9843          0.000000          O=C(OC(=O)C(F)(F)F)C(F)(F)F
#67868         1.991616              O=C(OCC(F)(F)F)C(F)(F)F
#100849        2.229154            C=CCOC(=O)C(F)(F)C(F)(F)F
#69614         2.272976         O=C(NOC(=O)C(F)(F)F)C(F)(F)F
#188429        2.317491          C=C(F)C(=O)OC(F)(F)C(C)(F)F
#137018        2.345655        CCOC(=O)C=C(C(F)(F)F)C(F)(F)F
#67908         2.387941              CCOC(=O)C(F)(F)C(F)(F)F
#67721         2.407293        COC(=O)C(F)(F)C(F)(F)C(F)(F)F
#67724         2.501380            C=CC(=O)OCC(F)(F)C(F)(F)F
#123488        2.563801         C=C(C)C(=O)OCC(F)(F)C(F)(F)F
#9642          2.587894       CCOC(=O)C(F)(F)C(F)(F)C(F)(F)F
#140676        2.592086             COC(=O)C(F)(F)C(F)=C(F)F
#9781          2.611027               COC(=O)C(F)(F)C(F)(F)F
#9775          2.624849         O=C(O)C(F)(F)C(F)(F)C(F)(F)F
#67807         2.668884    O=C(O)C(F)(F)C(F)(F)C(F)(F)C(=O)O
#67903         2.780984     C=CC(=O)OCC(F)(F)C(F)(F)C(F)(F)F
#75076         2.799258         C=CC(=O)OC(C(F)(F)F)C(F)(F)F
#164548        2.874228               FC(F)C(F)(F)OCC(F)(F)F
#84621         2.909806          O=C(C=C(O)C(F)(F)F)C(F)(F)F
#87125         2.924887           C=CC(=O)OCCC(F)(F)C(F)(F)F
#145376        2.934825         COC(=O)CC(=O)C(F)(F)C(F)(F)F
#81457         2.963362           C=COC(F)(F)C(F)(F)C(F)(F)F
#498102        2.986288               FC(=CC(F)(F)F)C(F)(F)F
#67772         2.986679          CC(=CC(=O)C(F)(F)F)C(F)(F)F
#173423        2.999769                  O=C(O)C(F)(F)OC(F)F
#83644         3.014887  C=C(C)C(=O)OCC(F)(F)C(F)(F)C(F)(F)F
#123089        3.021421  COC(=O)C(F)(F)C(F)(F)C(F)(F)C(=O)OC
#73686         3.023445          O=C(CC(=O)C(F)(F)F)C(F)(F)F
#9778          3.030964         O=C=NCC(F)(F)C(F)(F)C(F)(F)F
#92464         3.079639          FC(F)=C(F)OC(F)(F)C(F)(F)Cl







#%%
# step 9
# =============================================================================
# 此program最後一步，將30個similar data輸出成csv，之後可以直接用來預測，這邊需要放想要比較的target chemical在pubchem 2 million database裡的ID，(非pubchem上面的CID)
# =============================================================================

new_my_database_csv = pd.read_csv('1-2000000.csv')

similarData_original, similarData_clean = similarData_toDataFrame(new_my_database_csv, 9843)








































#%%

# =============================================================================
# # 找出距離最大的是誰
# =============================================================================

count = 0
for i, item in enumerate(totalDistance):
    if item > 10**90:
        count += 1
        print(i, 'and', item)
print(count)

#
#24757 and inf
#192518 and 3.0244512206244143e+103
#241916 and 7.040327597137338e+97
#451516 and 5.49788984236032e+98
#451517 and 2.4943062053961836e+99
#451518 and 4.963243595364703e+123
#451519 and 1.5615729098383056e+124
#466836 and 7.579995289653445e+102
#469768 and 1.4681118751668836e+133
#9


# 找出哪個dimension貢獻距離最大
count = 0
for i in range(2,len(new_my_database_csv.columns)):
    a = new_my_database_csv.iloc[9843, i]
    b = new_my_database_csv.iloc[192518, i]
    dis = (b-a)**2
    print(dis)
    count += dis
print(math.sqrt(count))   # 算距離
print(math.sqrt(count) == totalDistance[192518])

new_my_database_csv.iloc[9843,0:7]
new_my_database_csv.iloc[192518,0:7] # 可以發現是Ipc貢獻最大











#%%



