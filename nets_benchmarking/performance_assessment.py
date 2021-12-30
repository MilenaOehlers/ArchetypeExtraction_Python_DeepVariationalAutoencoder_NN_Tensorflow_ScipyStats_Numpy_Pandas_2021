import itertools as itt
from os import path
import pandas as pd
import numpy as np 
from scipy.stats import multivariate_normal
import os 
import pickle 
from collections import OrderedDict
#from func_collection import get_zfixed, points_in_simplex

def hellingerdistance(mus,sigmas,points_in_simplex):
    """Calculates metric between [0,1] of "distance" between two distributions (0->same distributions)
      mus: np.array: rows=dta pts, cols=coords
      sigmas= np.array: rows=dta pts, only 1 column
      points_in_simplex: calculated with same-named-function"""
      
    def tdistrsummed_in_simplex(mus=mus,sigmas=sigmas,points_in_simplex=points_in_simplex):
        print('here normally sigma error!')
        for i in range(mus.shape[0]):
            sigmas = np.abs(sigmas)
            try:
                multigauss = multivariate_normal(mus[i,:], np.diagflat(sigmas[i]))
            except:
                #print(mus[i,:],sigmas[i])
                multigauss = multivariate_normal(mus[i,:], np.diagflat(sigmas[i]))
            tdistr = multigauss.pdf(points_in_simplex)
            if i==0: tdistrsummed = tdistr
            if i>0:  tdistrsummed = tdistrsummed+tdistr
        tdistrsummed = tdistrsummed/np.sum(tdistrsummed) #integral over simplex is 1
        return tdistrsummed
    
    def uniform_in_simplex(points_in_simplex):
        n_points = points_in_simplex.shape[0]
        uniform = np.ones(n_points)/n_points # integral over simplex is one
        return uniform
    
    def gauss_in_simplex(points_in_simplex):
        dim_latent_space = points_in_simplex.shape[1]
        gauss = multivariate_normal(np.zeros(dim_latent_space,dtype=float), np.eye(dim_latent_space,dtype=float))
        simplexgauss = gauss.pdf(points_in_simplex) 
        simplexgauss = simplexgauss/np.sum(simplexgauss) #renormalizes gaussian so that integral over simplex is 1
        return simplexgauss
    all_t = tdistrsummed_in_simplex(mus,sigmas)
    bc_gauss = np.sum((all_t*gauss_in_simplex(points_in_simplex))**0.5)
    bc_uniform = np.sum((all_t*uniform_in_simplex(points_in_simplex))**0.5)
    H_gauss = (1-bc_gauss)**0.5
    H_uniform = (1-bc_uniform)**0.5
    return H_gauss, H_uniform #metric, scales between 0 (same distributions) and 1 (no distr overlap), fulfills triangle inequality

def hyperpar_comparison(list_of_resultfiles,points_in_simplex,dumpinto='hyp', overwrite=False, displayrounded=3):
    """- list_of_resultfiles must contain a collection of filenames; each filename indicates the hyperparameters 
    on which the collection of results contained in the file are based.
    The filename could exhibit the following form (as preset in daa_exe.ipynb):
    modelparams = [normalscores,version,at_loss_factor,target_loss_factor,recon_loss_factor,kl_loss_factor,anneal]
    res_filename = 'res_{}'.format(tuple(modelparams)) 
    - points_in_simplex must be calculated by same-named function
    - dumpinto is name of file in which resutls will be dumped, if already exists and overwrite=False: new filename = dumpinto + "1"
    - only if overwrite=True, file with name dumpinto will be overwritten if exists
    - with displayrounded = int, pickled results will stay non-rounded, but those returned by function will be."""
    res_path = "results_collections"
    if (overwrite is False and os.path.exists("{}/{}".format(res_path,dumpinto))):
        with open("{}/{}".format(res_path,dumpinto), 'rb') as pickled_results:
            hyperpar_df = pickle.load(pickled_results)
    else:
        hyperpar_df = pd.DataFrame(columns=['MSE_feats','MSE_targets','lat_gauss','lat_uniform','tot','err'])
        for filename in list_of_resultfiles:
            def load_results(filename):
                try:
                    with open("{}/{}".format(res_path,filename), 'rb') as pickled_results:
                        results = pickle.load(pickled_results)
                except:
                    print("{} is not contained in the reults_collections folder".format(filename))
                return results
            key = filename #[4:]
            results = load_results(filename)
            MSEin,MSEtar,latgauss,latuni = [],[],[],[]
            faulty = 0
            n_calcs = 0
            for _,item in results.items():
                n_calcs += 1
                mus,sigmas = item.at['latent space','features']
                if np.isnan(np.sum(mus)) or np.isnan(np.sum(sigmas)) or 0 in sigmas:
                    faulty += 1
                else: 
                    H_gauss, H_uniform = hellingerdistance(mus,sigmas,points_in_simplex)
                    MSEin = MSEin + [np.mean((item.at['real space','features'] - item.at['reconstructed real space','features'])**2)]
                    MSEtar = MSEtar + [np.mean((item.at['real space','target_color'] - item.at['reconstructed real space','target_color'])**2)]
                    latgauss = latgauss + [H_gauss]
                    latuni = latuni + [H_uniform]
            hyperpar_df.at[key,'MSE_feats'] = tuple([np.mean(MSEin),np.std(MSEin)])
            hyperpar_df.at[key,'MSE_targets'] = tuple([np.mean(MSEtar),np.std(MSEtar)]) 
            hyperpar_df.at[key,'lat_gauss'] = tuple([np.mean(latgauss),np.std(latgauss)])
            hyperpar_df.at[key,'lat_uniform'] = tuple([np.mean(latuni),np.std(latuni)])
            hyperpar_df.at[key,'tot'] = n_calcs
            hyperpar_df.at[key,'err'] = faulty
        if path.exists(dumpinto):
            dumpinto = dumpinto + '1'
        with open("{}/{}".format(res_path,dumpinto), 'wb') as file:
            pickle.dump(hyperpar_df, file)
    
    if isinstance(displayrounded,int):
        pd.set_option('display.expand_frame_repr', False)
        for ind in hyperpar_df.index:
            for col in ['MSE_feats','MSE_targets','lat_gauss','lat_uniform']:
                mean, stddev = hyperpar_df.at[ind,col][0],hyperpar_df.at[ind,col][1]
                hyperpar_df.at[ind,col] = tuple([mean.round(displayrounded),stddev.round(displayrounded)])
            hyperpar_df = hyperpar_df.rename(index={ind: "".join(ind.split(" "))})
    print('''-> the displayed tuple entries indicate the (mean, std.dev), respectively
-> MSE_feats and MSE_targets calculate the mean squared error for feature and target reconstruction
-> lat_gauss and lat_uniform compare the summed-up and normalized t-distributions to 
   standard normal gauss and uniform distribution in triangle, respectively, via hellinger distance.
   Hellinger distance is a metric ranging between 0 and 1; here: the smaller, the better.
-> tot indicates total number of calculations for resp. hyperpar. 
-> err indicates how many calculations for respective hyperparameters failed

Tuple entries in hyperpar_df stand for (mean,std.dev) respectively:''')
    print(hyperpar_df)
    return hyperpar_df
        
        



        