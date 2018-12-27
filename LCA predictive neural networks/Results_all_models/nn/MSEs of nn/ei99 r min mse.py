# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:48:06 2018

@author: e0348825
"""


'''ei99 resources'''


 relu, original x 
 the minimum mse is 0.0450 
-----
1 Layer, 50 Neurons w/ 
relu, original x, standardized y mse is 0.0450 
__________________________

 elu, original x 
 the minimum mse is 0.0478 
-----
3 Layer, 50 Neurons w/ 
elu, original x, standardized y mse is 0.0478 
__________________________

 relu, std x 
 the minimum mse is 0.1190 
-----
3 Layer, 50 Neurons w/ 
relu, standardized x, log1p y mse is 0.1190 
__________________________

 elu, std x 
 the minimum mse is 0.0387 
-----
3 Layer, 25 Neurons w/ 
elu, standardized x, log1p y mse is 0.0387 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.0307 
-----
2 Layer, 50 Neurons w/ 
relu, PCA (99%) x, standardized y mse is 0.0307 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.0186 
-----
3 Layer, 25 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.0186 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.2127 
-----
2 Layer, 50 Neurons w/ 
relu, PCA (80%) x, log1p y mse is 0.2127 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.1587 
-----
2 Layer, 50 Neurons w/ 
elu, PCA (80%) x, standardized y mse is 0.1587 
__________________________