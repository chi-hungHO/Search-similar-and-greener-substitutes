1. use "main_3rdPart" to import data from the folder [準備用來預測的data_剔除outlier]
		EI99 ecosystem vs descriptors_noOutlier.csv
		EI99 health vs descriptors_noOutlier.csv
		EI99 resourses vs descriptors_noOutlier.csv
		EI99 total vs descriptors_noOutlier.csv

		Recipe ecosystem vs descriptors_noOutlier.csv
		Recipe health vs descriptors_noOutlier.csv
		Recipe resources vs descriptors_noOutlier.csv
		Recipe total vs descriptors_noOutlier.csv

	this data are LCIA of 213 chemicals from Ecoinvent, which excludes 3 outliers from 216 chemicals.

2. use "main_3rdPart" and "function_get_prediction_fr_best_models.py" to predict  E H R T of EI99 and ReCiPe with best models


3.  use "main_3rdPart" & "function_get_mape_mse_n_scatter" to draw scatter plot


4. use "main_3rdPart" & "function_get_mape_mse_n_scatterRange" to narrow dowm the range


5. use "main_3rdPart" & "function_compare2_total" to compare EI99 total and EI99 e+h+r, to see which way of predicting total has better performance
	The results are presented as scatter graph
		this is also done to ReCiPe
		ps. e+h+r = total


6. use "main_3rdPart" to import data of 30 similar chemicals [chemical 9843 similarPoint_clean.csv] from the folder [Results_30_similarPoints]
		or on github, you can find [chemical 9843 similarPoint_clean.csv] in [Search-similar-and-greener-substitutes/Identification of similar chemicals/Results/]


7. for EI99 total, EI99 t has better performance than EI99 e+h+r
	so use "main_3rdPart" & "similarChemicals_ei99T" in "function_get_prediction_similar_chemicals" to predict EI99 t of 30 similar chemicals
	the prediction results are averaged for the results of 10 runs 
	also, the standard deviation of prediction of 10 times are presented


8. Similar to step 7, so use "main_3rdPart" & 
	"similarChemicals_recipeE"
	"similarChemicals_recipeH"
	"similarChemicals_recipeR" in "function_get_prediction_similar_chemicals"
		the prediction results are averaged for the results of 10 runs 
		also, the standard deviation of prediction of 10 times are presented


9. use "main_3rdPart" & [make_similar_prediction_into_graph] in [function_graph_select_greener_chemicals_version2] 
	to visualize trifluoroacetic anhydride & 30 similar chemicals的 EI99 and ReCiPe


10. use "main_3rdPart" to find 16 chemicals out of 30, whose EI99 and ReCiPe are both lower than trifluoroacetic anhydride 


11. use "main_3rdPart" & "function_graph_shows_greener_substitutes_boxplot" to visualize the EI99 of these 16 chemicals compared to other general chemicals

	
   