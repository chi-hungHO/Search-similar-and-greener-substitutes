# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:10:03 2018

@author: NUS_2
"""


'''
The results of ei99 e
'''
 relu, original x 
 the minimum mse is 0.0704 
-----
2 Layer, 50 Neurons w/ 
relu, original x, standardized y mse is 0.0704 
__________________________

 elu, original x 
 the minimum mse is 0.0877 
-----
1 Layer, 50 Neurons w/ 
elu, original x, original y mse is 0.0877 
__________________________

 relu, std x 
 the minimum mse is 0.8934 
-----
1 Layer, 25 Neurons w/ 
relu, standardized x, original y mse is 0.8934 
__________________________

 elu, std x 
 the minimum mse is 0.1006 
-----
2 Layer, 25 Neurons w/ 
elu, standardized x, log1p y mse is 0.1006 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.0653 
-----
2 Layer, 25 Neurons w/ 
relu, PCA (99%) x, standardized y mse is 0.0653 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.0320 
-----
2 Layer, 100 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.0320 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.0928 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (80%) x, original y mse is 0.0928 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.0919 
-----
2 Layer, 25 Neurons w/ 
elu, PCA (80%) x, original y mse is 0.0919 
__________________________