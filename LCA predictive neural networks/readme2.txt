


Preparation of data

1. Use Data [all data with all LCIA.xlsx] which includes all data from ecoinvent, in the folder [chemicals LCIA data]
	the data is here: https://bit.ly/2AkUcdk


2. Clean all the data, delete all the unwanted items (we want only chemicals), and delete unwanted LCIA (we want only EI99, ReCiPe)
	Use pubchem to find SMILES for each of 216 chemicals
	output: [IUPAC into smiles version10.csv] in the folder [chemicals LCIA data]


3. use "dataCrunch_collect 125 descriptors for 216 chemicals.py" to get CID from pubchem
	With CID, we can download the gz file of these 216 chemicals  
	(data is stored as [smiles_1_cid.sdf.gz] in the folder [chemicals LCIA data])
	with gz file, we can get 125 descriptors of these 216 chemicals by "create_a_chemi_mole_to_df.py"
	output [216 chemicals with descriptors.csv] in the folder [???????data]


4. use [216 chemicals with descriptors.csv] in the folder [???????data]
	separate them into EI99 E H R T, and ReCiPe E H R T
	(ReCiPe midpoints are not used this time)
	output  
		EI99 ecosystem vs descriptors.csv
		EI99 health vs descriptors.csv
		EI99 resourses vs descriptors.csv
		EI99 total vs descriptors.csv
	
		Recipe ecosystem vs descriptors.csv
		Recipe health vs descriptors.csv
		Recipe resources vs descriptors.csv
		Recipe total vs descriptors.csv


----------------------------------------------------------------------------------
predict by NN without k-fold 


5. use "main_version4.py" import data from

		EI99 ecosystem vs descriptors.csv
		EI99 health vs descriptors.csv
		EI99 resourses vs descriptors.csv
		EI99 total vs descriptors.csv
	
		Recipe ecosystem vs descriptors.csv
		Recipe health vs descriptors.csv
		Recipe resources vs descriptors.csv
		Recipe total vs descriptors.csv

6. use "main_version4.py" and "function_data_preprocessing" to do data preprocessing
	here we still need the function "function_get_data_treatedPCA", this can be found in the folder "Identification of similar chemicals"
	This will generate different x and y using 3 different split random seed

		'original x, original y'
		'original x, standardized y'
		'original x, log1p y'
		'standardized x, original y'
		'standardized x, standardized y'
		'standardized x, log1p y'
		'PCA (99%) x, original y'
		'PCA (99%) x, standardized y'
		'PCA (99%) x, log1p y'
		'PCA (80%) x, original y'
		'PCA (80%) x, standardized y'
		'PCA (80%) x, log1p y'

7. use "main_version4.py" & "function_best_nn_grid_search_version3" to generate all results (including different model structures)


8. use "main_version4.py" & "function_make_resultList_into_graph_version4" to visualize all results. For each impact category in EI99 & ReCiPe
	8 graphs would be generated


9. use "main_version4.py" & "function_graph_show_best_model_new" to visualize the best models


10. step 6-9 would be iterate for each impact category, therefore totally 8 impact categories would be predicted.

	




----------------------------------------------------------------------------------
predict by gradient boosting tryy


11. use "main_ensemble(XGB, LGBT)" import data from

		EI99 ecosystem vs descriptors.csv
		EI99 health vs descriptors.csv
		EI99 resourses vs descriptors.csv
		EI99 total vs descriptors.csv
	
		Recipe ecosystem vs descriptors.csv
		Recipe health vs descriptors.csv
		Recipe resources vs descriptors.csv
		Recipe total vs descriptors.csv


12. use "main_ensemble(XGB, LGBT)" & "function_best_ensemble_grid_search" to generate all results


13. use "main_ensemble(XGB, LGBT)" & "function_make_resultList_into_graph_ensemble" to visualize all results

