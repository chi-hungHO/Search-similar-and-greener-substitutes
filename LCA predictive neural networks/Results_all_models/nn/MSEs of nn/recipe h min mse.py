# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:21:58 2018

@author: NUS_2
"""



'''recipe h'''



 relu, original x 
 the minimum mse is 0.2077 
-----
2 Layer, 100 Neurons w/ 
relu, original x, standardized y mse is 0.2077 
__________________________

 elu, original x 
 the minimum mse is 0.1855 
-----
1 Layer, 250 Neurons w/ 
elu, original x, original y mse is 0.1855 
__________________________

 relu, std x 
 the minimum mse is 0.0793 
-----
2 Layer, 50 Neurons w/ 
relu, standardized x, original y mse is 0.0793 
__________________________

 elu, std x 
 the minimum mse is 0.1515 
-----
2 Layer, 25 Neurons w/ 
elu, standardized x, log1p y mse is 0.1515 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.1399 
-----
2 Layer, 25 Neurons w/ 
relu, PCA (99%) x, standardized y mse is 0.1399 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.0761 
-----
2 Layer, 25 Neurons w/ 
elu, PCA (99%) x, log1p y mse is 0.0761 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.2049 
-----
2 Layer, 250 Neurons w/ 
relu, PCA (80%) x, original y mse is 0.2049 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.1921 
-----
3 Layer, 25 Neurons w/ 
elu, PCA (80%) x, original y mse is 0.1921 
__________________________