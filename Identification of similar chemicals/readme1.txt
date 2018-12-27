!!!  Note: this section should be run under the rdkit environment  !!!  




1. First, use "function_create_a_chemi_mole_to_df"   convernt gz file downloaded from pubchem 
	at the same, generate descriptors for each chemical




#######  step 2-8 are just to compare the distance between 3 categories: original data, standardized data and PCA data
#######  Please just do step 9 to find similar chemicals, not necessary to do step2-8

2. use "main"   generate csv for the database obtained above


3. use "function_EucliDistance" to find similar points for original data


--------------------------------------------------------------------
4. use "function_standardize" to get standardized data (std_data)

5. use "function_EucliDistance" to find similar points for standardized data
--------------------------------------------------------------------




--------------------------------------------------------------------
6. use "function_get_data_treatedPCA" to find how many features to keep when using PCA

7. use "function_get_data_treatedPCA" to get PCA data

8. use "function_EucliDistance" to find similar points for PCA data
--------------------------------------------------------------------



9. the "function_similarData_toDataFrame" is a summary of functions above

	This step is to find the distance for chemicals after PCA only

	the input data is in the folder "pubchem full data" and the files such as "1-2000000.csv"
	
	what the function does is standardize and pca the data, then get the similar points with 125 descriptors
	then convert it into csv
	



	the output would be 2: clean data and original
	clean data: does not have SMILES [chemical 9843 similarPoint_clean.csv]
	original: have SMILES [chemical 9843 similarPoint_original.csv]

