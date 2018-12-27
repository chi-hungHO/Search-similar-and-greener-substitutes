# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:07:58 2018

@author: NUS_2
"""


'''recipe total'''


 relu, original x 
 the minimum mse is 0.5099 
-----
1 Layer, 100 Neurons w/ 
relu, original x, original y mse is 0.5099 
__________________________

 elu, original x 
 the minimum mse is 1.1813 
-----
1 Layer, 50 Neurons w/ 
elu, original x, standardized y mse is 1.1813 
__________________________

 relu, std x 
 the minimum mse is 0.8580 
-----
3 Layer, 25 Neurons w/ 
relu, standardized x, log1p y mse is 0.8580 
__________________________

 elu, std x 
 the minimum mse is 0.6675 
-----
3 Layer, 25 Neurons w/ 
elu, standardized x, original y mse is 0.6675 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.4660 
-----
2 Layer, 50 Neurons w/ 
relu, PCA (99%) x, original y mse is 0.4660 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.4393 
-----
2 Layer, 50 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.4393 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 1.4029 
-----
2 Layer, 25 Neurons w/ 
relu, PCA (80%) x, original y mse is 1.4029 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 1.1719 
-----
2 Layer, 50 Neurons w/ 
elu, PCA (80%) x, original y mse is 1.1719 
__________________________