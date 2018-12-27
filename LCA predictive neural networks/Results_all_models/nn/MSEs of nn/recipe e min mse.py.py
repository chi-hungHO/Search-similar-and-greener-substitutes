# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:44:25 2018

@author: NUS_2
"""




'''reciipe e'''



relu, original x 
 the minimum mse is 0.0303 
-----
1 Layer, 50 Neurons w/ 
relu, original x, standardized y mse is 0.0303 
__________________________

 elu, original x 
 the minimum mse is 0.0183 
-----
3 Layer, 25 Neurons w/ 
elu, original x, standardized y mse is 0.0183 
__________________________

 relu, std x 
 the minimum mse is 0.0515 
-----
2 Layer, 50 Neurons w/ 
relu, standardized x, standardized y mse is 0.0515 
__________________________

 elu, std x 
 the minimum mse is 0.0153 
-----
2 Layer, 25 Neurons w/ 
elu, standardized x, log1p y mse is 0.0153 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.0164 
-----
3 Layer, 25 Neurons w/ 
relu, PCA (99%) x, original y mse is 0.0164 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.0228 
-----
3 Layer, 25 Neurons w/ 
elu, PCA (99%) x, original y mse is 0.0228 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 0.0220 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (80%) x, standardized y mse is 0.0220 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 0.0182 
-----
2 Layer, 50 Neurons w/ 
elu, PCA (80%) x, standardized y mse is 0.0182 
__________________________
