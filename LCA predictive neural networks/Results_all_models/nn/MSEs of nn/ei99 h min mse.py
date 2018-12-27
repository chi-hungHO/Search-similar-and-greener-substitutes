# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:26:46 2018

@author: NUS_2
"""



'''
The results of ei99 h
'''
 relu, original x 
 the minimum mse is 0.5003 
-----
1 Layer, 250 Neurons w/ 
relu, original x, original y mse is 0.5003 
__________________________

 elu, original x 
 the minimum mse is 0.4705 
-----
1 Layer, 250 Neurons w/ 
elu, original x, log1p y mse is 0.4705 
__________________________

 relu, std x 
 the minimum mse is 0.2432 
-----
2 Layer, 50 Neurons w/ 
relu, standardized x, original y mse is 0.2432 
__________________________

 elu, std x 
 the minimum mse is 0.2284 
-----
1 Layer, 25 Neurons w/ 
elu, standardized x, log1p y mse is 0.2284 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.2424 
-----
2 Layer, 50 Neurons w/ 
relu, PCA (99%) x, log1p y mse is 0.2424 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.0809 
-----
2 Layer, 100 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.0809 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.5470 
-----
2 Layer, 50 Neurons w/ 
relu, PCA (80%) x, log1p y mse is 0.5470 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.4360 
-----
1 Layer, 50 Neurons w/ 
elu, PCA (80%) x, log1p y mse is 0.4360 
__________________________