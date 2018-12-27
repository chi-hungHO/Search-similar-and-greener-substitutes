# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:54:48 2018

@author: e0225113
"""


''' recipe r '''


 relu, original x 
 the minimum mse is 0.2841 
-----
3 Layer, 25 Neurons w/ 
relu, original x, standardized y mse is 0.2841 
__________________________

 elu, original x 
 the minimum mse is 0.3597 
-----
1 Layer, 250 Neurons w/ 
elu, original x, standardized y mse is 0.3597 
__________________________

 relu, std x 
 the minimum mse is 0.5104 
-----
1 Layer, 250 Neurons w/ 
relu, standardized x, standardized y mse is 0.5104 
__________________________

 elu, std x 
 the minimum mse is 0.1818 
-----
2 Layer, 50 Neurons w/ 
elu, standardized x, log1p y mse is 0.1818 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.1969 
-----
3 Layer, 25 Neurons w/ 
relu, PCA (99%) x, standardized y mse is 0.1969 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.1128 
-----
2 Layer, 25 Neurons w/ 
elu, PCA (99%) x, standardized y mse is 0.1128 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.2996 
-----
2 Layer, 100 Neurons w/ 
relu, PCA (80%) x, standardized y mse is 0.2996 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.6750 
-----
2 Layer, 100 Neurons w/ 
elu, PCA (80%) x, original y mse is 0.6750 
__________________________