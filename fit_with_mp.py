import os
#only print error messages from tensorflow
import tensorflow as tf
import numpy as np
import math
import time
import pickle
import bz2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
import copy
from sklearn.metrics import mean_squared_error,mean_absolute_error
import scipy.optimize as opt
from itertools import chain

class FitMP:

    def __init__(self):
        pass


    def analytical_eq(self,xy,xo,yo,No,T):   
        x, y = xy
        
        Nex = 100000-No
        
        out = (No/math.pi)*np.exp(-(x-xo)**2)*np.exp(-(y-yo)**2) + (Nex/(2*math.pi*T))*np.exp((-(x-xo)**2)/(2*T))*np.exp((-(y-yo)**2)/(2*T))

        # out = (out - np.min(out)) / np.max(out)
        return out.ravel()


    def get_performance_score(self,params,pair):

        x = np.linspace(-49, 50, 100)
        y = np.linspace(-49, 50, 100)
        x, y = np.meshgrid(x, y)
            
        n,t = params
        
        im,labs = pair

        performance = mean_absolute_error(self.analytical_eq((x,y),0,0,n,t),im.ravel())

        return performance
        
    def fit_preds_for_batch(self,x,y):
        
        new_preds = []

        for i,pair in enumerate(zip(x,y)):
            
            kwargs = {"bounds": [(0,100000),(60,400)], "args": list(pair)}

            res = opt.basinhopping(self.get_performance_score,x0 = (50000,175), minimizer_kwargs = kwargs)

            new_preds.append(res["x"][::-1])
            
        return np.array(new_preds)



    