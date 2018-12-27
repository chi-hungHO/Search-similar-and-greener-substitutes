# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:05:50 2018

@author: NUS_2
"""


''' ei99 human health'''
 relu, original x 
 the minimum mse is 0.5019 
-----
1 Layer, 250 Neurons w/ 
relu, original x, standardized y mse is 0.5019 
__________________________

 elu, original x 
 the minimum mse is 0.1599 
-----
1 Layer, 100 Neurons w/ 
elu, original x, log1p y mse is 0.1599 
__________________________

 relu, std x 
 the minimum mse is 0.2158 
-----
1 Layer, 100 Neurons w/ 
relu, standardized x, log1p y mse is 0.2158 
__________________________

 elu, std x 
 the minimum mse is 0.0903 
-----
2 Layer, 50 Neurons w/ 
elu, standardized x, original y mse is 0.0903 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.3505 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (99%) x, original y mse is 0.3505 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.1207 
-----
2 Layer, 100 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.1207 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.5340 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (80%) x, standardized y mse is 0.5340 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.4185 
-----
1 Layer, 250 Neurons w/ 
elu, PCA (80%) x, log1p y mse is 0.4185 
__________________________