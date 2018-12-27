# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:40:18 2018

@author: e0348825
"""



''' ei99 total'''


 relu, original x 
 the minimum mse is 0.8306 
-----
1 Layer, 250 Neurons w/ 
relu, original x, standardized y mse is 0.8306 
__________________________

 elu, original x 
 the minimum mse is 1.2456 
-----
1 Layer, 50 Neurons w/ 
elu, original x, original y mse is 1.2456 
__________________________

 relu, std x 
 the minimum mse is 0.4715 
-----
1 Layer, 25 Neurons w/ 
relu, standardized x, log1p y mse is 0.4715 
__________________________

 elu, std x 
 the minimum mse is 0.6562 
-----
2 Layer, 50 Neurons w/ 
elu, standardized x, log1p y mse is 0.6562 
__________________________

 relu, PCA (99%) x 
 the minimum mse is 0.9774 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (99%) x, log1p y mse is 0.9774 
__________________________

 elu, PCA (99%) x 
 the minimum mse is 0.3781 
-----
1 Layer, 25 Neurons w/ 
elu, PCA (99%) x, log1p y mse is 0.3781 
__________________________

 relu, PCA (80%) x 
 the minimum mse is 1.5587 
-----
3 Layer, 50 Neurons w/ 
relu, PCA (80%) x, log1p y mse is 1.5587 
__________________________

 elu, PCA (80%) x 
 the minimum mse is 1.9395 
-----
2 Layer, 50 Neurons w/ 
elu, PCA (80%) x, log1p y mse is 1.9395 
__________________________