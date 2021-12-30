#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:31:54 2020

@author: oehlers
"""
import numpy as np
import pandas as pd
from os import path
import itertools as itt

def get_zfixed ( dim_latent_space ):
            
    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2
  
        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
                    
    return z_fixed

def points_in_simplex(zfixed,fineness):
    """Calculates positions of equally distributed points in simplex, 
    whose distance to their neighbors decreases with fineness."""
    dim_latent_space = zfixed.shape[1]
    
    res_path = "results_collections"
    filename = 'points_in_triangle({},{})'.format(dim_latent_space,fineness)
    
    if path.exists("{}/{}".format(res_path,filename)):
        points = np.array(pd.read_csv("{}/{}".format(res_path,filename)))
    
    else:
        
        def weights(dim_latent_space=dim_latent_space,fineness=fineness):
            x = np.expand_dims(np.linspace(0,1,fineness+1),axis=1)
            A = np.array([[0]*(dim_latent_space+1)])
            for i in list(itt.product(*[x]*(dim_latent_space+1)))[1:]:
                if np.sum(np.array(i))<=1:
                    A = np.concatenate((A,np.array(i).T),axis=0)
            return A

        points = np.matmul(weights(),zfixed)
        pd.DataFrame(points).to_csv("{}/{}".format(res_path,filename), index=False)
    
    return points

# written for data creation and separation, but not necessary in the end
# might still be useful for future applications:
def alsotuples_firstarg(func):
    def newfunc(*args):
        if isinstance(args[0],tuple):
            disentangled = [[*[args[0][i],*args[1:]]] for i in range(len(args[0]))]
            return tuple([func(*i) for i in disentangled])
        else: return func(*args)
    return newfunc