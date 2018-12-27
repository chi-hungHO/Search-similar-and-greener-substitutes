# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:56:36 2018

@author: e0225113
"""


#%%
# =============================================================================
# 這個function是用來將pubchem的gz壓縮檔變成pd dataframe，要變成CSV要自己另外to.cvs
# =============================================================================

from __future__ import print_function




def create_a_chemi_mole_to_df(data):
    
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdMolDescriptors
    
    
    
    import os
    import numpy as np
    import pandas as pd
    
    import gzip
    import pickle

    

    os.getcwd()

    my_zip = gzip.open(data)
    my_dezip = Chem.ForwardSDMolSupplier(my_zip)

#    my_dezip_data = [x for x in my_dezip if x is not None ]
    # 這邊選擇使用pickle，因為pickle檔案比較小，程式可以跑得比較快
    my_dezip_data = [pickle.loads(pickle.dumps(x)) for x in my_dezip if x is not None]










# 將全部的function儲存在字典裏面
    functions = {}
    f_i = 0
    my_columns = []
# =============================================================================
# get SMILE name
# =============================================================================
    def get_smile(my_dezip_data):
        
        my_smiles = []
        for molecule in my_dezip_data:
            my_smiles += [Chem.MolToSmiles(molecule)]
        return my_smiles
#    smiles = get_smile(my_dezip_data)
    functions[f_i] = get_smile
    my_columns += [str(f_i) + ',  ' + 'SMILE']
    f_i += 1
    
# =============================================================================
# get the number of non-H atom
# =============================================================================
    def get_NumAtoms(my_dezip_data):
        
        my_AtomNum_no_H = []
        my_H_Num = []
        
        for molecule in my_dezip_data:
            my_AtomNum_no_H += [molecule.GetNumAtoms()]
            
            molecule_2 = Chem.AddHs(molecule)
            my_H_Num += [(molecule_2.GetNumAtoms() - molecule.GetNumAtoms())]
        
        return my_AtomNum_no_H
#    Num_atom_not_h, Num_atom_h = get_NumAtoms(my_dezip_data)
    functions[f_i] = get_NumAtoms
    my_columns += [str(f_i) + ',  ' + 'AtomNum_no_H']
    f_i += 1

# =============================================================================
# # get the number of H atom
# =============================================================================
    
    def get_Num_H(my_dezip_data):
        
        my_AtomNum_no_H = []
        my_H_Num = []
        
        for molecule in my_dezip_data:
            my_AtomNum_no_H += [molecule.GetNumAtoms()]
            
            molecule_2 = Chem.AddHs(molecule)
            my_H_Num += [(molecule_2.GetNumAtoms() - molecule.GetNumAtoms())]
        
        return my_H_Num
#    Num_atom_not_h, Num_atom_h = get_NumAtoms(my_dezip_data)
    functions[f_i] = get_Num_H
    my_columns += [str(f_i) + ',  ' + 'AtomNum_H']
    f_i += 1
# =============================================================================
# 得到descriptor BalabanJ
# =============================================================================
    def get_BalabanJ(my_dezip_data):
        
        my_BalabanJ = []
        for molecule in my_dezip_data:
            my_BalabanJ += [Descriptors.BalabanJ(molecule)]
        return my_BalabanJ
    functions[f_i] = get_BalabanJ
    my_columns += [str(f_i) + ',  ' + 'BalabanJ']
    f_i += 1
# =============================================================================
# 得到descriptor BertzCT
# =============================================================================
    def get_BertzCT(my_dezip_data):
        
        my_BertzCT = []
        for molecule in my_dezip_data:
            my_BertzCT += [Descriptors.BertzCT(molecule)]
        return my_BertzCT
    functions[f_i] = get_BertzCT
    my_columns += [str(f_i) + ',  ' + 'BertzCT']
    f_i += 1

# =============================================================================
# 得到descriptor Ipc
# =============================================================================
    def get_Ipc(my_dezip_data):
        
        my_Ipc = []
        for molecule in my_dezip_data:
            my_Ipc += [Descriptors.Ipc(molecule)]
        return my_Ipc
    functions[f_i] = get_Ipc
    my_columns += [str(f_i) + ',  ' + 'Ipc']
    f_i += 1
# =============================================================================
# 得到descriptor HallKierAlpha
# =============================================================================
    def get_HallKierAlpha(my_dezip_data):
        
        my_HallKierAlpha = []
        for molecule in my_dezip_data:
            my_HallKierAlpha += [Descriptors.HallKierAlpha(molecule)]
        return my_HallKierAlpha
    functions[f_i] = get_HallKierAlpha
    my_columns += [str(f_i) + ',  ' + 'HallKierAlpha']
    f_i += 1
# =============================================================================
# 得到descriptor Kappa1
# =============================================================================
    def get_Kappa1(my_dezip_data):
        
        my_Kappa1 = []
        for molecule in my_dezip_data:
            my_Kappa1 += [Descriptors.Kappa1(molecule)]
        return my_Kappa1
    functions[f_i] = get_Kappa1
    my_columns += [str(f_i) + ',  ' + 'Kappa1']
    f_i += 1
# =============================================================================
# 得到descriptor Kappa2
# =============================================================================
    def get_Kappa2(my_dezip_data):
        
        my_Kappa2 = []
        for molecule in my_dezip_data:
            my_Kappa2 += [Descriptors.Kappa2(molecule)]
        return my_Kappa2
    functions[f_i] = get_Kappa2
    my_columns += [str(f_i) + ',  ' + 'Kappa2']
    f_i += 1
# =============================================================================
# 得到descriptor Kappa3
# =============================================================================
    def get_Kappa3(my_dezip_data):
        
        my_Kappa3 = []
        for molecule in my_dezip_data:
            my_Kappa3 += [Descriptors.Kappa3(molecule)]
        return my_Kappa3
    functions[f_i] = get_Kappa3
    my_columns += [str(f_i) + ',  ' + 'Kappa3']
    f_i += 1

# =============================================================================
# 得到descriptor Chi0
# =============================================================================
    def get_Chi0(my_dezip_data):
        
        my_Chi0 = []
        for molecule in my_dezip_data:
            my_Chi0 += [Descriptors.Chi0(molecule)]
        return my_Chi0
    functions[f_i] = get_Chi0
    my_columns += [str(f_i) + ',  ' + 'Chi0']
    f_i += 1
# =============================================================================
# 得到descriptor Chi0n
# =============================================================================
    def get_Chi0n(my_dezip_data):
        
        my_Chi0n = []
        for molecule in my_dezip_data:
            my_Chi0n += [Descriptors.Chi0n(molecule)]
        return my_Chi0n
    functions[f_i] = get_Chi0n
    my_columns += [str(f_i) + ',  ' + 'Chi0n']
    f_i += 1
# =============================================================================
# 得到descriptor Chi0v
# =============================================================================
    def get_Chi0v(my_dezip_data):
        
        my_Chi0v = []
        for molecule in my_dezip_data:
            my_Chi0v += [Descriptors.Chi0v(molecule)]
        return my_Chi0v
    functions[f_i] = get_Chi0v
    my_columns += [str(f_i) + ',  ' + 'Chi0v']
    f_i += 1
    
# =============================================================================
# 得到descriptor Chi1
# =============================================================================
    def get_Chi1(my_dezip_data):
        
        my_Chi1 = []
        for molecule in my_dezip_data:
            my_Chi1 += [Descriptors.Chi1(molecule)]
        return my_Chi1
    functions[f_i] = get_Chi1
    my_columns += [str(f_i) + ',  ' + 'Chi1']
    f_i += 1
# =============================================================================
# 得到descriptor Chi1n
# =============================================================================
    def get_Chi1n(my_dezip_data):
        
        my_Chi1n = []
        for molecule in my_dezip_data:
            my_Chi1n += [Descriptors.Chi1n(molecule)]
        return my_Chi1n
    functions[f_i] = get_Chi1n
    my_columns += [str(f_i) + ',  ' + 'Chi1n']
    f_i += 1
# =============================================================================
# 得到descriptor Chi1v
# =============================================================================
    def get_Chi1v(my_dezip_data):
        
        my_Chi1v = []
        for molecule in my_dezip_data:
            my_Chi1v += [Descriptors.Chi1v(molecule)]
        return my_Chi1v
    functions[f_i] = get_Chi1v
    my_columns += [str(f_i) + ',  ' + 'Chi1v']
    f_i += 1
# =============================================================================
# 得到descriptor Chi2n
# =============================================================================
    def get_Chi2n(my_dezip_data):
        
        my_Chi2n = []
        for molecule in my_dezip_data:
            my_Chi2n += [Descriptors.Chi2n(molecule)]
        return my_Chi2n
    functions[f_i] = get_Chi2n
    my_columns += [str(f_i) + ',  ' + 'Chi2n']
    f_i += 1
# =============================================================================
# 得到descriptor Chi2v
# =============================================================================
    def get_Chi2v(my_dezip_data):
        
        my_Chi2v = []
        for molecule in my_dezip_data:
            my_Chi2v += [Descriptors.Chi2v(molecule)]
        return my_Chi2v
    functions[f_i] = get_Chi2v
    my_columns += [str(f_i) + ',  ' + 'Chi2v']
    f_i += 1
# =============================================================================
# 得到descriptor Chi3n
# =============================================================================
    def get_Chi3n(my_dezip_data):
        
        my_Chi3n = []
        for molecule in my_dezip_data:
            my_Chi3n += [Descriptors.Chi3n(molecule)]
        return my_Chi3n
    functions[f_i] = get_Chi3n
    my_columns += [str(f_i) + ',  ' + 'Chi3n']
    f_i += 1
# =============================================================================
# 得到descriptor Chi3v
# =============================================================================
    def get_Chi3v(my_dezip_data):
        
        my_Chi3v = []
        for molecule in my_dezip_data:
            my_Chi3v += [Descriptors.Chi3v(molecule)]
        return my_Chi3v
    functions[f_i] = get_Chi3v
    my_columns += [str(f_i) + ',  ' + 'Chi3v']
    f_i += 1
# =============================================================================
# 得到descriptor Chi4n
# =============================================================================
    def get_Chi4n(my_dezip_data):
        
        my_Chi4n = []
        for molecule in my_dezip_data:
            my_Chi4n += [Descriptors.Chi4n(molecule)]
        return my_Chi4n
    functions[f_i] = get_Chi4n
    my_columns += [str(f_i) + ',  ' + 'Chi4n']
    f_i += 1

# =============================================================================
# 得到descriptor Chi4v
# =============================================================================
    def get_Chi4v(my_dezip_data):
        
        my_Chi4v = []
        for molecule in my_dezip_data:
            my_Chi4v += [Descriptors.Chi4v(molecule)]
        return my_Chi4v
    functions[f_i] = get_Chi4v
    my_columns += [str(f_i) + ',  ' + 'Chi4v']
    f_i += 1
# =============================================================================
# 得到descriptor MolLogP
# =============================================================================
    def get_MolLogP(my_dezip_data):
        
        my_MolLogP = []
        for molecule in my_dezip_data:
            my_MolLogP += [Descriptors.MolLogP(molecule)]
        return my_MolLogP
    functions[f_i] = get_MolLogP
    my_columns += [str(f_i) + ',  ' + 'MolLogP']
    f_i += 1
                       
# =============================================================================
# 得到descriptor MolMR
# =============================================================================
    def get_MolMR(my_dezip_data):
        
        my_MolMR = []
        for molecule in my_dezip_data:
            my_MolMR += [Descriptors.MolMR(molecule)]
        return my_MolMR
    functions[f_i] = get_MolMR
    my_columns += [str(f_i) + ',  ' + 'MolMR']
    f_i += 1
# =============================================================================
# 得到descriptor MolWt
# =============================================================================
    def get_MolWt(my_dezip_data):
        
        my_MolWt = []
        for molecule in my_dezip_data:
            my_MolWt += [Descriptors.MolWt(molecule)]
        return my_MolWt
    functions[f_i] = get_MolWt
    my_columns += [str(f_i) + ',  ' + 'MolWt']
    f_i += 1
                       
# =============================================================================
# 得到descriptor ExactMolWt
# =============================================================================
    def get_ExactMolWt(my_dezip_data):
        
        my_ExactMolWt = []
        for molecule in my_dezip_data:
            my_ExactMolWt += [Descriptors.ExactMolWt(molecule)]
        return my_ExactMolWt
    functions[f_i] = get_ExactMolWt
    my_columns += [str(f_i) + ',  ' + 'ExactMolWt']
    f_i += 1
                       
# =============================================================================
# 得到descriptor HeavyAtomCount
# =============================================================================
    def get_HeavyAtomCount(my_dezip_data):
        
        my_HeavyAtomCount = []
        for molecule in my_dezip_data:
            my_HeavyAtomCount += [Descriptors.HeavyAtomCount(molecule)]
        return my_HeavyAtomCount
    functions[f_i] = get_HeavyAtomCount
    my_columns += [str(f_i) + ',  ' + 'HeavyAtomCount']
    f_i += 1
                       
# =============================================================================
# 得到descriptor HeavyAtomMolWt
# =============================================================================
    def get_HeavyAtomMolWt(my_dezip_data):
        
        my_HeavyAtomMolWt = []
        for molecule in my_dezip_data:
            my_HeavyAtomMolWt += [Descriptors.HeavyAtomMolWt(molecule)]
        return my_HeavyAtomMolWt
    functions[f_i] = get_HeavyAtomMolWt
    my_columns += [str(f_i) + ',  ' + 'HeavyAtomMolWt']
    f_i += 1
                       
# =============================================================================
# 得到descriptor NHOHCount
# =============================================================================
    def get_NHOHCount(my_dezip_data):
        
        my_NHOHCount = []
        for molecule in my_dezip_data:
            my_NHOHCount += [Descriptors.NHOHCount(molecule)]
        return my_NHOHCount
    functions[f_i] = get_NHOHCount
    my_columns += [str(f_i) + ',  ' + 'NHOHCount']
    f_i += 1
                       
# =============================================================================
# 得到descriptor NOCount
# =============================================================================
    def get_NOCount(my_dezip_data):
        
        my_NOCount = []
        for molecule in my_dezip_data:
            my_NOCount += [Descriptors.NOCount(molecule)]
        return my_NOCount
    functions[f_i] = get_NOCount
    my_columns += [str(f_i) + ',  ' + 'NOCount']
    f_i += 1
# =============================================================================
# 得到descriptor NumHAcceptors
# =============================================================================
    def get_NumHAcceptors(my_dezip_data):
        
        my_NumHAcceptors = []
        for molecule in my_dezip_data:
            my_NumHAcceptors += [Descriptors.NumHAcceptors(molecule)]
        return my_NumHAcceptors
    functions[f_i] = get_NumHAcceptors
    my_columns += [str(f_i) + ',  ' + 'NumHAcceptors']
    f_i += 1
# =============================================================================
# 得到descriptor 
# =============================================================================
    def get_NumHDonors(my_dezip_data):
        
        my_NumHDonors = []
        for molecule in my_dezip_data:
            my_NumHDonors += [Descriptors.NumHDonors(molecule)]
        return my_NumHDonors
    functions[f_i] = get_NumHDonors
    my_columns += [str(f_i) + ',  ' + 'NumHDonors']
    f_i += 1
# =============================================================================
# 得到descriptor NumHeteroatoms
# =============================================================================
    def get_NumHeteroatoms(my_dezip_data):
        
        my_NumHeteroatoms = []
        for molecule in my_dezip_data:
            my_NumHeteroatoms += [Descriptors.NumHeteroatoms(molecule)]
        return my_NumHeteroatoms
    functions[f_i] = get_NumHeteroatoms
    my_columns += [str(f_i) + ',  ' + 'NumHeteroatoms']
    f_i += 1
# =============================================================================
# 得到descriptor NumRotatableBonds
# =============================================================================
    def get_NumRotatableBonds(my_dezip_data):
        
        my_NumRotatableBonds = []
        for molecule in my_dezip_data:
            my_NumRotatableBonds += [Descriptors.NumRotatableBonds(molecule)]
        return my_NumRotatableBonds
    functions[f_i] = get_NumRotatableBonds
    my_columns += [str(f_i) + ',  ' + 'NumRotatableBonds']
    f_i += 1
# =============================================================================
# 得到descriptor NumValenceElectrons
# =============================================================================
    def get_NumValenceElectrons(my_dezip_data):
        
        my_NumValenceElectrons = []
        for molecule in my_dezip_data:
            my_NumValenceElectrons += [Descriptors.NumValenceElectrons(molecule)]
        return my_NumValenceElectrons
    functions[f_i] = get_NumValenceElectrons
    my_columns += [str(f_i) + ',  ' + 'NumValenceElectrons']
    f_i += 1

# =============================================================================
# 得到descriptor CalcNumAmideBonds
# =============================================================================
    def get_CalcNumAmideBonds(my_dezip_data):
        
        my_CalcNumAmideBonds = []
        for molecule in my_dezip_data:
            my_CalcNumAmideBonds += [rdMolDescriptors.CalcNumAmideBonds(molecule)]
        return my_CalcNumAmideBonds
    functions[f_i] = get_CalcNumAmideBonds
    my_columns += [str(f_i) + ',  ' + 'CalcNumAmideBonds']
    f_i += 1

# =============================================================================
# 得到descriptor CalcNumAromaticRings
# =============================================================================
    def get_CalcNumAromaticRings(my_dezip_data):
        
        my_CalcNumAromaticRings = []
        for molecule in my_dezip_data:
            my_CalcNumAromaticRings += [rdMolDescriptors.CalcNumAromaticRings(molecule)]
        return my_CalcNumAromaticRings
    functions[f_i] = get_CalcNumAromaticRings
    my_columns += [str(f_i) + ',  ' + 'CalcNumAromaticRings']
    f_i += 1
# =============================================================================
# 得到descriptor CalcNumRings
# =============================================================================
    def get_CalcNumRings(my_dezip_data):
        
        my_CalcNumRings = []
        for molecule in my_dezip_data:
            my_CalcNumRings += [rdMolDescriptors.CalcNumRings(molecule)]
        return my_CalcNumRings
    functions[f_i] = get_CalcNumRings
    my_columns += [str(f_i) + ',  ' + 'CalcNumRings']
    f_i += 1
# =============================================================================
# 得到descriptor CalcNumSaturatedRings
# =============================================================================
    def get_CalcNumSaturatedRings(my_dezip_data):
        
        my_CalcNumSaturatedRings = []
        for molecule in my_dezip_data:
            my_CalcNumSaturatedRings += [rdMolDescriptors.CalcNumSaturatedRings(molecule)]
        return my_CalcNumSaturatedRings
    functions[f_i] = get_CalcNumSaturatedRings
    my_columns += [str(f_i) + ',  ' + 'CalcNumSaturatedRings']
    f_i += 1
    
# =============================================================================
# 得到descriptor CalcNumAliphaticRings
# =============================================================================
    def get_CalcNumAliphaticRings(my_dezip_data):
        
        my_CalcNumAliphaticRings = []
        for molecule in my_dezip_data:
            my_CalcNumAliphaticRings += [rdMolDescriptors.CalcNumAliphaticRings(molecule)]
        return my_CalcNumAliphaticRings
    functions[f_i] = get_CalcNumAliphaticRings
    my_columns += [str(f_i) + ',  ' + 'CalcNumAliphaticRings']
    f_i += 1
    
# =============================================================================
# 得到descriptor CalcNumAromaticCarbocycles
# =============================================================================
    def get_CalcNumAromaticCarbocycles(my_dezip_data):
        
        my_CalcNumAromaticCarbocycles = []
        for molecule in my_dezip_data:
            my_CalcNumAromaticCarbocycles += [rdMolDescriptors.CalcNumAromaticCarbocycles(molecule)]
        return my_CalcNumAromaticCarbocycles
    functions[f_i] = get_CalcNumAromaticCarbocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumAromaticCarbocycles']
    f_i += 1
# =============================================================================
# 得到descriptor CalcNumAromaticHeterocycles
# =============================================================================
    def get_CalcNumAromaticHeterocycles(my_dezip_data):
        
        my_CalcNumAromaticHeterocycles = []
        for molecule in my_dezip_data:
            my_CalcNumAromaticHeterocycles += [rdMolDescriptors.CalcNumAromaticHeterocycles(molecule)]
        return my_CalcNumAromaticHeterocycles
    functions[f_i] = get_CalcNumAromaticHeterocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumAromaticHeterocycles']
    f_i += 1
    
   # =============================================================================
# 得到descriptor CalcNumHeterocycles
# =============================================================================
    def get_CalcNumHeterocycles(my_dezip_data):
        
        my_CalcNumHeterocycles = []
        for molecule in my_dezip_data:
            my_CalcNumHeterocycles += [rdMolDescriptors.CalcNumHeterocycles(molecule)]
        return my_CalcNumHeterocycles
    functions[f_i] = get_CalcNumHeterocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumHeterocycles']
    f_i += 1
    
   # =============================================================================
# 得到descriptor CalcNumSaturatedCarbocycles
# =============================================================================
    def get_CalcNumSaturatedCarbocycles(my_dezip_data):
        
        my_CalcNumSaturatedCarbocycles = []
        for molecule in my_dezip_data:
            my_CalcNumSaturatedCarbocycles += [rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule)]
        return my_CalcNumSaturatedCarbocycles
    functions[f_i] = get_CalcNumSaturatedCarbocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumSaturatedCarbocycles']
    f_i += 1
    
    
   # =============================================================================
# 得到descriptor CalcNumSaturatedHeterocycles
# =============================================================================
    def get_CalcNumSaturatedHeterocycles(my_dezip_data):
        
        my_CalcNumSaturatedHeterocycles = []
        for molecule in my_dezip_data:
            my_CalcNumSaturatedHeterocycles += [rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule)]
        return my_CalcNumSaturatedHeterocycles
    functions[f_i] = get_CalcNumSaturatedHeterocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumSaturatedHeterocycles']
    f_i += 1
    
# =============================================================================
# 得到descriptor CalcNumAliphaticCarbocycles
# =============================================================================
    def get_CalcNumAliphaticCarbocycles(my_dezip_data):
        
        my_CalcNumAliphaticCarbocycles = []
        for molecule in my_dezip_data:
            my_CalcNumAliphaticCarbocycles += [rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule)]
        return my_CalcNumAliphaticCarbocycles
    functions[f_i] = get_CalcNumAliphaticCarbocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumAliphaticCarbocycles']
    f_i += 1
    
# =============================================================================
# 得到descriptor CalcNumAliphaticHeterocycles
# =============================================================================
    def get_CalcNumAliphaticHeterocycles(my_dezip_data):
        
        my_CalcNumAliphaticHeterocycles = []
        for molecule in my_dezip_data:
            my_CalcNumAliphaticHeterocycles += [rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule)]
        return my_CalcNumAliphaticHeterocycles
    functions[f_i] = get_CalcNumAliphaticHeterocycles
    my_columns += [str(f_i) + ',  ' + 'CalcNumAliphaticHeterocycles']
    f_i += 1
                           
# =============================================================================
# 得到descriptor RingCount
# =============================================================================
    def get_RingCount(my_dezip_data):
        
        my_RingCount = []
        for molecule in my_dezip_data:
            my_RingCount += [Descriptors.RingCount(molecule)]
        return my_RingCount
    functions[f_i] = get_RingCount
    my_columns += [str(f_i) + ',  ' + 'RingCount']
    f_i += 1
# =============================================================================
# 得到descriptor FractionCSP3
# =============================================================================
    def get_FractionCSP3(my_dezip_data):
        
        my_FractionCSP3 = []
        for molecule in my_dezip_data:
            my_FractionCSP3 += [Descriptors.FractionCSP3(molecule)]
        return my_FractionCSP3
    functions[f_i] = get_FractionCSP3
    my_columns += [str(f_i) + ',  ' + 'FractionCSP3']
    f_i += 1
# =============================================================================
# 得到descriptor CalcNumSpiroAtoms
# =============================================================================
    def get_CalcNumSpiroAtoms(my_dezip_data):
        
        my_CalcNumSpiroAtoms = []
        for molecule in my_dezip_data:
            my_CalcNumSpiroAtoms += [rdMolDescriptors.CalcNumSpiroAtoms(molecule)]
        return my_CalcNumSpiroAtoms
    functions[f_i] = get_CalcNumSpiroAtoms
    my_columns += [str(f_i) + ',  ' + 'CalcNumSpiroAtoms']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcNumBridgeheadAtoms
# =============================================================================
    def get_CalcNumBridgeheadAtoms(my_dezip_data):
        
        my_CalcNumBridgeheadAtoms = []
        for molecule in my_dezip_data:
            my_CalcNumBridgeheadAtoms += [rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)]
        return my_CalcNumBridgeheadAtoms
    functions[f_i] = get_CalcNumBridgeheadAtoms
    my_columns += [str(f_i) + ',  ' + 'CalcNumBridgeheadAtoms']
    f_i += 1
# =============================================================================
# 得到descriptor TPSA
# =============================================================================
    def get_TPSA(my_dezip_data):
        
        my_TPSA = []
        for molecule in my_dezip_data:
            my_TPSA += [Descriptors.TPSA(molecule)]
        return my_TPSA
    functions[f_i] = get_TPSA
    my_columns += [str(f_i) + ',  ' + 'TPSA']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor LabuteASA
# =============================================================================
    def get_LabuteASA(my_dezip_data):
        
        my_LabuteASA = []
        for molecule in my_dezip_data:
            my_LabuteASA += [Descriptors.LabuteASA(molecule)]
        return my_LabuteASA
    functions[f_i] = get_LabuteASA
    my_columns += [str(f_i) + ',  ' + 'LabuteASA']
    f_i += 1                             
    
    
# =============================================================================
# 得到descriptor PEOE_VSA1
# =============================================================================
    def get_PEOE_VSA1(my_dezip_data):
        
        my_PEOE_VSA1 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA1 += [Descriptors.PEOE_VSA1(molecule)]
        return my_PEOE_VSA1
    functions[f_i] = get_PEOE_VSA1
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA1']
    f_i += 1                                     
                                                      
# =============================================================================
# 得到descriptor PEOE_VSA2
# =============================================================================
    def get_PEOE_VSA2(my_dezip_data):
        
        my_PEOE_VSA2 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA2 += [Descriptors.PEOE_VSA2(molecule)]
        return my_PEOE_VSA2
    functions[f_i] = get_PEOE_VSA2
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA2']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA3
# =============================================================================
    def get_PEOE_VSA3(my_dezip_data):
        
        my_PEOE_VSA3 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA3 += [Descriptors.PEOE_VSA3(molecule)]
        return my_PEOE_VSA3
    functions[f_i] = get_PEOE_VSA3
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA3']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA4
# =============================================================================
    def get_PEOE_VSA4(my_dezip_data):
        
        my_PEOE_VSA4 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA4 += [Descriptors.PEOE_VSA4(molecule)]
        return my_PEOE_VSA4
    functions[f_i] = get_PEOE_VSA4
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA4']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA5
# =============================================================================
    def get_PEOE_VSA5(my_dezip_data):
        
        my_PEOE_VSA5 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA5 += [Descriptors.PEOE_VSA5(molecule)]
        return my_PEOE_VSA5
    functions[f_i] = get_PEOE_VSA5
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA5']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA6
# =============================================================================
    def get_PEOE_VSA6(my_dezip_data):
        
        my_PEOE_VSA6 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA6 += [Descriptors.PEOE_VSA6(molecule)]
        return my_PEOE_VSA6
    functions[f_i] = get_PEOE_VSA6
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA6']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA7
# =============================================================================
    def get_PEOE_VSA7(my_dezip_data):
        
        my_PEOE_VSA7 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA7 += [Descriptors.PEOE_VSA7(molecule)]
        return my_PEOE_VSA7
    functions[f_i] = get_PEOE_VSA7
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA7']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor PEOE_VSA8
# =============================================================================
    def get_PEOE_VSA8(my_dezip_data):
        
        my_PEOE_VSA8 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA8 += [Descriptors.PEOE_VSA8(molecule)]
        return my_PEOE_VSA8
    functions[f_i] = get_PEOE_VSA8
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA8']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor PEOE_VSA9
# =============================================================================
    def get_PEOE_VSA9(my_dezip_data):
        
        my_PEOE_VSA9 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA9 += [Descriptors.PEOE_VSA9(molecule)]
        return my_PEOE_VSA9
    functions[f_i] = get_PEOE_VSA9
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA9']
    f_i += 1                                     
# =============================================================================
# 得到descriptor PEOE_VSA10
# =============================================================================
    def get_PEOE_VSA10(my_dezip_data):
        
        my_PEOE_VSA10 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA10 += [Descriptors.PEOE_VSA10(molecule)]
        return my_PEOE_VSA10
    functions[f_i] = get_PEOE_VSA10
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA10']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor PEOE_VSA11
# =============================================================================
    def get_PEOE_VSA11(my_dezip_data):
        
        my_PEOE_VSA11 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA11 += [Descriptors.PEOE_VSA11(molecule)]
        return my_PEOE_VSA11
    functions[f_i] = get_PEOE_VSA11
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA11']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor PEOE_VSA12
# =============================================================================
    def get_PEOE_VSA12(my_dezip_data):
        
        my_PEOE_VSA12 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA12 += [Descriptors.PEOE_VSA12(molecule)]
        return my_PEOE_VSA12
    functions[f_i] = get_PEOE_VSA12
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA12']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor PEOE_VSA13
# =============================================================================
    def get_PEOE_VSA13(my_dezip_data):
        
        my_PEOE_VSA13 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA13 += [Descriptors.PEOE_VSA13(molecule)]
        return my_PEOE_VSA13
    functions[f_i] = get_PEOE_VSA13
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA13']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor PEOE_VSA14
# =============================================================================
    def get_PEOE_VSA14(my_dezip_data):
        
        my_PEOE_VSA14 = []
        for molecule in my_dezip_data:
            my_PEOE_VSA14 += [Descriptors.PEOE_VSA14(molecule)]
        return my_PEOE_VSA14
    functions[f_i] = get_PEOE_VSA14
    my_columns += [str(f_i) + ',  ' + 'PEOE_VSA14']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor SMR_VSA1
# =============================================================================
    def get_SMR_VSA1(my_dezip_data):
        
        my_SMR_VSA1 = []
        for molecule in my_dezip_data:
            my_SMR_VSA1 += [Descriptors.SMR_VSA1(molecule)]
        return my_SMR_VSA1
    functions[f_i] = get_SMR_VSA1
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA1']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor SMR_VSA2
# =============================================================================
    def get_SMR_VSA2(my_dezip_data):
        
        my_SMR_VSA2 = []
        for molecule in my_dezip_data:
            my_SMR_VSA2 += [Descriptors.SMR_VSA2(molecule)]
        return my_SMR_VSA2
    functions[f_i] = get_SMR_VSA2
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA2']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor SMR_VSA3
# =============================================================================
    def get_SMR_VSA3(my_dezip_data):
        
        my_SMR_VSA3 = []
        for molecule in my_dezip_data:
            my_SMR_VSA3 += [Descriptors.SMR_VSA3(molecule)]
        return my_SMR_VSA3
    functions[f_i] = get_SMR_VSA3
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA3']
    f_i += 1                                     
                                                         
# =============================================================================
# 得到descriptor SMR_VSA4
# =============================================================================
    def get_SMR_VSA4(my_dezip_data):
        
        my_SMR_VSA4 = []
        for molecule in my_dezip_data:
            my_SMR_VSA4 += [Descriptors.SMR_VSA4(molecule)]
        return my_SMR_VSA4
    functions[f_i] = get_SMR_VSA4
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA4']
    f_i += 1                                     

    
    
# =============================================================================
# 得到descriptor SMR_VSA5
# =============================================================================
    def get_SMR_VSA5(my_dezip_data):
        
        my_SMR_VSA5 = []
        for molecule in my_dezip_data:
            my_SMR_VSA5 += [Descriptors.SMR_VSA5(molecule)]
        return my_SMR_VSA5
    functions[f_i] = get_SMR_VSA5
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA5']
    f_i += 1                                     

                                                        
# =============================================================================
# 得到descriptor SMR_VSA6
# =============================================================================
    def get_SMR_VSA6(my_dezip_data):
        
        my_SMR_VSA6 = []
        for molecule in my_dezip_data:
            my_SMR_VSA6 += [Descriptors.SMR_VSA6(molecule)]
        return my_SMR_VSA6
    functions[f_i] = get_SMR_VSA6
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA6']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SMR_VSA7
# =============================================================================
    def get_SMR_VSA7(my_dezip_data):
        
        my_SMR_VSA7 = []
        for molecule in my_dezip_data:
            my_SMR_VSA7 += [Descriptors.SMR_VSA7(molecule)]
        return my_SMR_VSA7
    functions[f_i] = get_SMR_VSA7
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA7']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SMR_VSA8
# =============================================================================
    def get_SMR_VSA8(my_dezip_data):
        
        my_SMR_VSA8 = []
        for molecule in my_dezip_data:
            my_SMR_VSA8 += [Descriptors.SMR_VSA8(molecule)]
        return my_SMR_VSA8
    functions[f_i] = get_SMR_VSA8
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA8']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SMR_VSA9
# =============================================================================
    def get_SMR_VSA9(my_dezip_data):
        
        my_SMR_VSA9 = []
        for molecule in my_dezip_data:
            my_SMR_VSA9 += [Descriptors.SMR_VSA9(molecule)]
        return my_SMR_VSA9
    functions[f_i] = get_SMR_VSA9
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA9']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SMR_VSA10
# =============================================================================
    def get_SMR_VSA10(my_dezip_data):
        
        my_SMR_VSA10 = []
        for molecule in my_dezip_data:
            my_SMR_VSA10 += [Descriptors.SMR_VSA10(molecule)]
        return my_SMR_VSA10
    functions[f_i] = get_SMR_VSA10
    my_columns += [str(f_i) + ',  ' + 'SMR_VSA10']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA1
# =============================================================================
    def get_SlogP_VSA1(my_dezip_data):
        
        my_SlogP_VSA1 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA1 += [Descriptors.SlogP_VSA1(molecule)]
        return my_SlogP_VSA1
    functions[f_i] = get_SlogP_VSA1
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA1']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA2
# =============================================================================
    def get_SlogP_VSA2(my_dezip_data):
        
        my_SlogP_VSA2 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA2 += [Descriptors.SlogP_VSA2(molecule)]
        return my_SlogP_VSA2
    functions[f_i] = get_SlogP_VSA2
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA2']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA3
# =============================================================================
    def get_SlogP_VSA3(my_dezip_data):
        
        my_SlogP_VSA3 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA3 += [Descriptors.SlogP_VSA3(molecule)]
        return my_SlogP_VSA3
    functions[f_i] = get_SlogP_VSA3
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA3']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor SlogP_VSA4
# =============================================================================
    def get_SlogP_VSA4(my_dezip_data):
        
        my_SlogP_VSA4 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA4 += [Descriptors.SlogP_VSA4(molecule)]
        return my_SlogP_VSA4
    functions[f_i] = get_SlogP_VSA4
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA4']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA5
# =============================================================================
    def get_SlogP_VSA5(my_dezip_data):
        
        my_SlogP_VSA5 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA5 += [Descriptors.SlogP_VSA5(molecule)]
        return my_SlogP_VSA5
    functions[f_i] = get_SlogP_VSA5
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA5']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA6
# =============================================================================
    def get_SlogP_VSA6(my_dezip_data):
        
        my_SlogP_VSA6 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA6 += [Descriptors.SlogP_VSA6(molecule)]
        return my_SlogP_VSA6
    functions[f_i] = get_SlogP_VSA6
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA6']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA7
# =============================================================================
    def get_SlogP_VSA7(my_dezip_data):
        
        my_SlogP_VSA7 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA7 += [Descriptors.SlogP_VSA7(molecule)]
        return my_SlogP_VSA7
    functions[f_i] = get_SlogP_VSA7
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA7']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA8
# =============================================================================
    def get_SlogP_VSA8(my_dezip_data):
        
        my_SlogP_VSA8 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA8 += [Descriptors.SlogP_VSA8(molecule)]
        return my_SlogP_VSA8
    functions[f_i] = get_SlogP_VSA8
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA8']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA9
# =============================================================================
    def get_SlogP_VSA9(my_dezip_data):
        
        my_SlogP_VSA9 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA9 += [Descriptors.SlogP_VSA9(molecule)]
        return my_SlogP_VSA9
    functions[f_i] = get_SlogP_VSA9
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA9']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA10
# =============================================================================
    def get_SlogP_VSA10(my_dezip_data):
        
        my_SlogP_VSA10 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA10 += [Descriptors.SlogP_VSA10(molecule)]
        return my_SlogP_VSA10
    functions[f_i] = get_SlogP_VSA10
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA10']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA11
# =============================================================================
    def get_SlogP_VSA11(my_dezip_data):
        
        my_SlogP_VSA11 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA11 += [Descriptors.SlogP_VSA11(molecule)]
        return my_SlogP_VSA11
    functions[f_i] = get_SlogP_VSA11
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA11']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor SlogP_VSA12
# =============================================================================
    def get_SlogP_VSA12(my_dezip_data):
        
        my_SlogP_VSA12 = []
        for molecule in my_dezip_data:
            my_SlogP_VSA12 += [Descriptors.SlogP_VSA12(molecule)]
        return my_SlogP_VSA12
    functions[f_i] = get_SlogP_VSA12
    my_columns += [str(f_i) + ',  ' + 'SlogP_VSA12']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor EState_VSA1
# =============================================================================
    def get_EState_VSA1(my_dezip_data):
        
        my_EState_VSA1 = []
        for molecule in my_dezip_data:
            my_EState_VSA1 += [Descriptors.EState_VSA1(molecule)]
        return my_EState_VSA1
    functions[f_i] = get_EState_VSA1
    my_columns += [str(f_i) + ',  ' + 'EState_VSA1']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor EState_VSA2
# =============================================================================
    def get_EState_VSA2(my_dezip_data):
        
        my_EState_VSA2 = []
        for molecule in my_dezip_data:
            my_EState_VSA2 += [Descriptors.EState_VSA2(molecule)]
        return my_EState_VSA2
    functions[f_i] = get_EState_VSA2
    my_columns += [str(f_i) + ',  ' + 'EState_VSA2']
    f_i += 1                                     
                                                        
# =============================================================================
# 得到descriptor EState_VSA3
# =============================================================================
    def get_EState_VSA3(my_dezip_data):
        
        my_EState_VSA3 = []
        for molecule in my_dezip_data:
            my_EState_VSA3 += [Descriptors.EState_VSA3(molecule)]
        return my_EState_VSA3
    functions[f_i] = get_EState_VSA3
    my_columns += [str(f_i) + ',  ' + 'EState_VSA3']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA4
# =============================================================================
    def get_EState_VSA4(my_dezip_data):
        
        my_EState_VSA4 = []
        for molecule in my_dezip_data:
            my_EState_VSA4 += [Descriptors.EState_VSA4(molecule)]
        return my_EState_VSA4
    functions[f_i] = get_EState_VSA4
    my_columns += [str(f_i) + ',  ' + 'EState_VSA4']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA5
# =============================================================================
    def get_EState_VSA5(my_dezip_data):
        
        my_EState_VSA5 = []
        for molecule in my_dezip_data:
            my_EState_VSA5 += [Descriptors.EState_VSA5(molecule)]
        return my_EState_VSA5
    functions[f_i] = get_EState_VSA5
    my_columns += [str(f_i) + ',  ' + 'EState_VSA5']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA6
# =============================================================================
    def get_EState_VSA6(my_dezip_data):
        
        my_EState_VSA6 = []
        for molecule in my_dezip_data:
            my_EState_VSA6 += [Descriptors.EState_VSA6(molecule)]
        return my_EState_VSA6
    functions[f_i] = get_EState_VSA6
    my_columns += [str(f_i) + ',  ' + 'EState_VSA6']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA7
# =============================================================================
    def get_EState_VSA7(my_dezip_data):
        
        my_EState_VSA7 = []
        for molecule in my_dezip_data:
            my_EState_VSA7 += [Descriptors.EState_VSA7(molecule)]
        return my_EState_VSA7
    functions[f_i] = get_EState_VSA7
    my_columns += [str(f_i) + ',  ' + 'EState_VSA7']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA8
# =============================================================================
    def get_EState_VSA8(my_dezip_data):
        
        my_EState_VSA8 = []
        for molecule in my_dezip_data:
            my_EState_VSA8 += [Descriptors.EState_VSA8(molecule)]
        return my_EState_VSA8
    functions[f_i] = get_EState_VSA8
    my_columns += [str(f_i) + ',  ' + 'EState_VSA8']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA9
# =============================================================================
    def get_EState_VSA9(my_dezip_data):
        
        my_EState_VSA9 = []
        for molecule in my_dezip_data:
            my_EState_VSA9 += [Descriptors.EState_VSA9(molecule)]
        return my_EState_VSA9
    functions[f_i] = get_EState_VSA9
    my_columns += [str(f_i) + ',  ' + 'EState_VSA9']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA10
# =============================================================================
    def get_EState_VSA10(my_dezip_data):
        
        my_EState_VSA10 = []
        for molecule in my_dezip_data:
            my_EState_VSA10 += [Descriptors.EState_VSA10(molecule)]
        return my_EState_VSA10
    functions[f_i] = get_EState_VSA10
    my_columns += [str(f_i) + ',  ' + 'EState_VSA10']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor EState_VSA11
# =============================================================================
    def get_EState_VSA11(my_dezip_data):
        
        my_EState_VSA11 = []
        for molecule in my_dezip_data:
            my_EState_VSA11 += [Descriptors.EState_VSA11(molecule)]
        return my_EState_VSA11
    functions[f_i] = get_EState_VSA11
    my_columns += [str(f_i) + ',  ' + 'EState_VSA11']
    f_i += 1                                     
# =============================================================================
# 得到descriptor VSA_EState1
# =============================================================================
    def get_VSA_EState1(my_dezip_data):
        
        my_VSA_EState1 = []
        for molecule in my_dezip_data:
            my_VSA_EState1 += [Descriptors.VSA_EState1(molecule)]
        return my_VSA_EState1
    functions[f_i] = get_VSA_EState1
    my_columns += [str(f_i) + ',  ' + 'VSA_EState1']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor VSA_EState2
# =============================================================================
    def get_VSA_EState2(my_dezip_data):
        
        my_VSA_EState2 = []
        for molecule in my_dezip_data:
            my_VSA_EState2 += [Descriptors.VSA_EState2(molecule)]
        return my_VSA_EState2
    functions[f_i] = get_VSA_EState2
    my_columns += [str(f_i) + ',  ' + 'VSA_EState2']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor VSA_EState3
# =============================================================================
    def get_VSA_EState3(my_dezip_data):
        
        my_VSA_EState3 = []
        for molecule in my_dezip_data:
            my_VSA_EState3 += [Descriptors.VSA_EState3(molecule)]
        return my_VSA_EState3
    functions[f_i] = get_VSA_EState3
    my_columns += [str(f_i) + ',  ' + 'VSA_EState3']
    f_i += 1                                     
    
# =============================================================================
# 得到descriptor VSA_EState4
# =============================================================================
    def get_VSA_EState4(my_dezip_data):
        
        my_VSA_EState4 = []
        for molecule in my_dezip_data:
            my_VSA_EState4 += [Descriptors.VSA_EState4(molecule)]
        return my_VSA_EState4
    functions[f_i] = get_VSA_EState4
    my_columns += [str(f_i) + ',  ' + 'VSA_EState4']
    f_i += 1                                     
    
    
    
# =============================================================================
# 得到descriptor VSA_EState5
# =============================================================================
    def get_VSA_EState5(my_dezip_data):
        
        my_VSA_EState5 = []
        for molecule in my_dezip_data:
            my_VSA_EState5 += [Descriptors.VSA_EState5(molecule)]
        return my_VSA_EState5
    functions[f_i] = get_VSA_EState5
    my_columns += [str(f_i) + ',  ' + 'VSA_EState5']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor VSA_EState6
# =============================================================================
    def get_VSA_EState6(my_dezip_data):
        
        my_VSA_EState6 = []
        for molecule in my_dezip_data:
            my_VSA_EState6 += [Descriptors.VSA_EState6(molecule)]
        return my_VSA_EState6
    functions[f_i] = get_VSA_EState6
    my_columns += [str(f_i) + ',  ' + 'VSA_EState6']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor VSA_EState7
# =============================================================================
    def get_VSA_EState7(my_dezip_data):
        
        my_VSA_EState7 = []
        for molecule in my_dezip_data:
            my_VSA_EState7 += [Descriptors.VSA_EState7(molecule)]
        return my_VSA_EState7
    functions[f_i] = get_VSA_EState7
    my_columns += [str(f_i) + ',  ' + 'VSA_EState7']
    f_i += 1                                     
    
    
    
# =============================================================================
# 得到descriptor VSA_EState8
# =============================================================================
    def get_VSA_EState8(my_dezip_data):
        
        my_VSA_EState8 = []
        for molecule in my_dezip_data:
            my_VSA_EState8 += [Descriptors.VSA_EState8(molecule)]
        return my_VSA_EState8
    functions[f_i] = get_VSA_EState8
    my_columns += [str(f_i) + ',  ' + 'VSA_EState8']
    f_i += 1                                     
    
    
# =============================================================================
# 得到descriptor VSA_EState9
# =============================================================================
    def get_VSA_EState9(my_dezip_data):
        
        my_VSA_EState9 = []
        for molecule in my_dezip_data:
            my_VSA_EState9 += [Descriptors.VSA_EState9(molecule)]
        return my_VSA_EState9
    functions[f_i] = get_VSA_EState9
    my_columns += [str(f_i) + ',  ' + 'VSA_EState9']
    f_i += 1                                     

# =============================================================================
# 得到descriptor VSA_EState10
# =============================================================================
    def get_VSA_EState10(my_dezip_data):
        
        my_VSA_EState10 = []
        for molecule in my_dezip_data:
            my_VSA_EState10 += [Descriptors.VSA_EState10(molecule)]
        return my_VSA_EState10
    functions[f_i] = get_VSA_EState10
    my_columns += [str(f_i) + ',  ' + 'VSA_EState10']
    f_i += 1                                     
                      
# =============================================================================
# 得到descriptor CalcPBF
# =============================================================================
    def get_CalcPBF(my_dezip_data):
        
        my_CalcPBF = []
        for molecule in my_dezip_data:
            my_CalcPBF += [rdMolDescriptors.CalcPBF(molecule)]
        return my_CalcPBF
    functions[f_i] = get_CalcPBF
    my_columns += [str(f_i) + ',  ' + 'CalcPBF']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcPMI1
# =============================================================================
    def get_CalcPMI1(my_dezip_data):
        
        my_CalcPMI1 = []
        for molecule in my_dezip_data:
            my_CalcPMI1 += [rdMolDescriptors.CalcPMI1(molecule)]
        return my_CalcPMI1
    functions[f_i] = get_CalcPMI1
    my_columns += [str(f_i) + ',  ' + 'CalcPMI1']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor CalcPMI2
# =============================================================================
    def get_CalcPMI2(my_dezip_data):
        
        my_CalcPMI2 = []
        for molecule in my_dezip_data:
            my_CalcPMI2 += [rdMolDescriptors.CalcPMI2(molecule)]
        return my_CalcPMI2
    functions[f_i] = get_CalcPMI2
    my_columns += [str(f_i) + ',  ' + 'CalcPMI2']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcPMI3
# =============================================================================
    def get_CalcPMI3(my_dezip_data):
        
        my_CalcPMI3 = []
        for molecule in my_dezip_data:
            my_CalcPMI3 += [rdMolDescriptors.CalcPMI3(molecule)]
        return my_CalcPMI3
    functions[f_i] = get_CalcPMI3
    my_columns += [str(f_i) + ',  ' + 'CalcPMI3']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcNPR1
# =============================================================================
    def get_CalcNPR1(my_dezip_data):
        
        my_CalcNPR1 = []
        for molecule in my_dezip_data:
            my_CalcNPR1 += [rdMolDescriptors.CalcNPR1(molecule)]
        return my_CalcNPR1
    functions[f_i] = get_CalcNPR1
    my_columns += [str(f_i) + ',  ' + 'CalcNPR1']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor CalcNPR2
# =============================================================================
    def get_CalcNPR2(my_dezip_data):
        
        my_CalcNPR2 = []
        for molecule in my_dezip_data:
            my_CalcNPR2 += [rdMolDescriptors.CalcNPR2(molecule)]
        return my_CalcNPR2
    functions[f_i] = get_CalcNPR2
    my_columns += [str(f_i) + ',  ' + 'CalcNPR2']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor CalcRadiusOfGyration
# =============================================================================
    def get_CalcRadiusOfGyration(my_dezip_data):
        
        my_CalcRadiusOfGyration = []
        for molecule in my_dezip_data:
            my_CalcRadiusOfGyration += [rdMolDescriptors.CalcRadiusOfGyration(molecule)]
        return my_CalcRadiusOfGyration
    functions[f_i] = get_CalcRadiusOfGyration
    my_columns += [str(f_i) + ',  ' + 'CalcRadiusOfGyration']
    f_i += 1
    
    
    
    
# =============================================================================
# 得到descriptor CalcInertialShapeFactor
# =============================================================================
    def get_CalcInertialShapeFactor(my_dezip_data):
        
        my_CalcInertialShapeFactor = []
        for molecule in my_dezip_data:
            my_CalcInertialShapeFactor += [rdMolDescriptors.CalcInertialShapeFactor(molecule)]
        return my_CalcInertialShapeFactor
    functions[f_i] = get_CalcInertialShapeFactor
    my_columns += [str(f_i) + ',  ' + 'CalcInertialShapeFactor']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcEccentricity
# =============================================================================
    def get_CalcEccentricity(my_dezip_data):
        
        my_CalcEccentricity = []
        for molecule in my_dezip_data:
            my_CalcEccentricity += [rdMolDescriptors.CalcEccentricity(molecule)]
        return my_CalcEccentricity
    functions[f_i] = get_CalcEccentricity
    my_columns += [str(f_i) + ',  ' + 'CalcEccentricity']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcAsphericity
# =============================================================================
    def get_CalcAsphericity(my_dezip_data):
        
        my_CalcAsphericity = []
        for molecule in my_dezip_data:
            my_CalcAsphericity += [rdMolDescriptors.CalcAsphericity(molecule)]
        return my_CalcAsphericity
    functions[f_i] = get_CalcAsphericity
    my_columns += [str(f_i) + ',  ' + 'CalcAsphericity']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor CalcSpherocityIndex
# =============================================================================
    def get_CalcSpherocityIndex(my_dezip_data):
        
        my_CalcSpherocityIndex = []
        for molecule in my_dezip_data:
            my_CalcSpherocityIndex += [rdMolDescriptors.CalcSpherocityIndex(molecule)]
        return my_CalcSpherocityIndex
    functions[f_i] = get_CalcSpherocityIndex
    my_columns += [str(f_i) + ',  ' + 'CalcSpherocityIndex']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcNumHBA
# =============================================================================
    def get_CalcNumHBA(my_dezip_data):
        
        my_CalcNumHBA = []
        for molecule in my_dezip_data:
            my_CalcNumHBA += [rdMolDescriptors.CalcNumHBA(molecule)]
        return my_CalcNumHBA
    functions[f_i] = get_CalcNumHBA
    my_columns += [str(f_i) + ',  ' + 'CalcNumHBA']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcNumHBD
# =============================================================================
    def get_CalcNumHBD(my_dezip_data):
        
        my_CalcNumHBD = []
        for molecule in my_dezip_data:
            my_CalcNumHBD += [rdMolDescriptors.CalcNumHBD(molecule)]
        return my_CalcNumHBD
    functions[f_i] = get_CalcNumHBD
    my_columns += [str(f_i) + ',  ' + 'CalcNumHBD']
    f_i += 1
    
    
    
# =============================================================================
# 得到descriptor CalcNumLipinskiHBA
# =============================================================================
    def get_CalcNumLipinskiHBA(my_dezip_data):
        
        my_CalcNumLipinskiHBA = []
        for molecule in my_dezip_data:
            my_CalcNumLipinskiHBA += [rdMolDescriptors.CalcNumLipinskiHBA(molecule)]
        return my_CalcNumLipinskiHBA
    functions[f_i] = get_CalcNumLipinskiHBA
    my_columns += [str(f_i) + ',  ' + 'CalcNumLipinskiHBA']
    f_i += 1
    
    
# =============================================================================
# 得到descriptor CalcNumLipinskiHBD
# =============================================================================
    def get_CalcNumLipinskiHBD(my_dezip_data):
        
        my_CalcNumLipinskiHBD = []
        for molecule in my_dezip_data:
            my_CalcNumLipinskiHBD += [rdMolDescriptors.CalcNumLipinskiHBD(molecule)]
        return my_CalcNumLipinskiHBD
    functions[f_i] = get_CalcNumLipinskiHBD
    my_columns += [str(f_i) + ',  ' + 'CalcNumLipinskiHBD']
    f_i += 1
                                                                                                           
                
                                                                     
                                                                                                                                          

## =============================================================================
## 得到descriptor MaxAbsPartialCharge
## =============================================================================
#    def get_MaxAbsPartialCharge(my_dezip_data):
#        
#        my_MaxAbsPartialCharge = []
#        for molecule in my_dezip_data:
#            my_MaxAbsPartialCharge += [Descriptors.MaxAbsPartialCharge(molecule)]
#        return my_MaxAbsPartialCharge
#    functions[f_i] = get_MaxAbsPartialCharge
#    my_columns += [str(f_i) + ',  ' + 'MaxAbsPartialCharge']
#    f_i += 1                     
#
## =============================================================================
## 得到descriptor MaxPartialCharge
## =============================================================================
#    def get_MaxPartialCharge(my_dezip_data):
#        
#        my_MaxPartialCharge = []
#        for molecule in my_dezip_data:
#            my_MaxPartialCharge += [Descriptors.MaxPartialCharge(molecule)]
#        return my_MaxPartialCharge
#    functions[f_i] = get_MaxPartialCharge
#    my_columns += [str(f_i) + ',  ' + 'MaxPartialCharge']
#    f_i += 1  
#                   
## =============================================================================
## 得到descriptor MinAbsPartialCharge
## =============================================================================
#    def get_MinAbsPartialCharge(my_dezip_data):
#        
#        my_MinAbsPartialCharge = []
#        for molecule in my_dezip_data:
#            my_MinAbsPartialCharge += [Descriptors.MinAbsPartialCharge(molecule)]
#        return my_MinAbsPartialCharge
#    functions[f_i] = get_MinAbsPartialCharge
#    my_columns += [str(f_i) + ',  ' + 'MinAbsPartialCharge']
#    f_i += 1                 
#    
## =============================================================================
## 得到descriptor MinPartialCharge
## =============================================================================
#    def get_MinPartialCharge(my_dezip_data):
#        
#        my_MinPartialCharge = []
#        for molecule in my_dezip_data:
#            my_MinPartialCharge += [Descriptors.MinPartialCharge(molecule)]
#        return my_MinPartialCharge
#    functions[f_i] = get_MinPartialCharge
#    my_columns += [str(f_i) + ',  ' + 'MinPartialCharge']
#    f_i += 1                     
#
#      
# =============================================================================
# 得到descriptor NumRadicalElectrons
# =============================================================================
    def get_NumRadicalElectrons(my_dezip_data):
        
        my_NumRadicalElectrons = []
        for molecule in my_dezip_data:
            my_NumRadicalElectrons += [Descriptors.NumRadicalElectrons(molecule)]
        return my_NumRadicalElectrons
    functions[f_i] = get_NumRadicalElectrons
    my_columns += [str(f_i) + ',  ' + 'NumRadicalElectrons']
    f_i += 1                     





















    
# =============================================================================
# 創造dataframe
# =============================================================================

#    my_columns = [x for x in range(500)]
    zero_df = pd.DataFrame({'to_be_eliminated': np.ones(len(my_dezip_data))})
    print(zero_df.columns,'\n\n\n')
    
    for i, columns in zip(functions, my_columns):
        df = pd.DataFrame({ columns : functions[i](my_dezip_data)})
        zero_df = pd.concat([zero_df, df], axis = 1)
    
    
    
    
    
    for i,j in zip(functions, my_columns):
        print('function is ', i, ', columns is ', j)

    
    return zero_df

    


# =============================================================================
# 得到final database
# =============================================================================


