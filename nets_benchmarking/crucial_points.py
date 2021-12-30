#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:52:41 2020

@author: oehlers
"""
import random
import itertools as itt

def divide(divfun,threshold):
    def exe(arr,*otherarrs_samedivide_lst):
        #divfun,threshold = divfun_threshold
        group1 = [divfun(arr,i)<=threshold for i in range(arr.shape[0])]
        group2 = [not truefalse for truefalse in group1]
        group1tup = tuple([arr[group1],*[otherarr[group1] for otherarr in otherarrs_samedivide_lst]])
        group2tup = tuple([arr[group2],*[otherarr[group2] for otherarr in otherarrs_samedivide_lst]])
        return group1tup, group2tup
    return exe

# =============================================================================
# if divlat_fun_threshold is not None:
#     assert isinstance(divlat_fun_threshold,tuple) and callable(divlat_fun_threshold[0]) and isinstance(divlat_fun_threshold[1],float)
#     group1tup, group2tup = divide(*divlat_fun_threshold)(simplex_points,[real_space,target])
#     return group1tup, group2tup
# else: 
# =============================================================================
def divrandom(*arrs):
    lenn = arrs[0].shape[0]
    rand1 = [bool(random.choice([True, False])) for i in range(lenn)]
    rand2 = [not truefalse for truefalse in rand1]
    return tuple([arr[rand1] for arr in arrs]), tuple([arr[rand2] for arr in arrs])

def get_groupdicts(data,divfun,threshold):
    groups, newdicts = {},{}
    for d,t in itt.product(['div','rand'],['train','test']):
        if d=='div': func = divide(divfun,threshold)
        if d=='rand': func = divrandom
        groups[t+d+'1'],groups[t+d+'2'] = func(data[t+'_simplex'],data[t+'_feat'],data[t+'_targets'])
        
    for t in ['train','test']: groups[t+'both'] = tuple(data[key] for key in [t+'_simplex',t+'_feat',t+'_targets'])
    
    allgroups = ['div1','rand1','div2','rand2','both']
    for traingroup,testgroup in itt.product(allgroups,allgroups):
        newdicts[traingroup+testgroup] = {'train_simplex':groups['train'+traingroup][0],'train_feat':groups['train'+traingroup][1],'train_targets':groups['train'+traingroup][2],
                        'test_simplex':groups['test'+testgroup][0],'test_feat':groups['test'+testgroup][1],'test_targets':groups['test'+testgroup][2]}
# =============================================================================
#     #only for testing:
#     for key, dictt in newdicts.items():
#         print(key)
#         for key2, value in dictt.items():
#             print(key2,value.shape)
# =============================================================================
    return newdicts