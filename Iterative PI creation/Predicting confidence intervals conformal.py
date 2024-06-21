# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 08:56:06 2024

@author: dan
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 07:45:40 2024

@author: dan
"""
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMRegressor
#import lightgbm as lgb
# from scikeras.wrappers import KerasRegressor

from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import (regression_coverage_score, regression_mean_width_score, regression_ssc,regression_ssc_score)
from mapie.subsample import Subsample
from mapie.conformity_scores import GammaConformityScore,ResidualNormalisedScore

from scipy import stats
from scipy.stats import norm, halfnorm
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea

import random
import time
import numpy as np
import pandas as pd

#import relevant code
import generate_samples as gen
import plotting_functions as plotting
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point" # for CQR sklearn model

out_file_name = "results.csv"

np.random.seed(42)

gamma_score = 0
real_data = 0
if real_data:
    simulate_data = 0
else:
    simulate_data = 1

rng = np.random.RandomState(42) #Set seed
alpha = 0.1
np.random.seed(42)
"""
Available prediction methods:
-----------------------------
naive
jackknife
jackknife_plus
jackknife_minmax
cv
cv_plus
cv_minmax
jackknife_plus_ab
jackknife_minmax_ab
conformalized_quantile_regression
The ensemble batch prediction intervals (EnbPI) method
"""

#%%
# =============================================================================
# Prediction and training
# =============================================================================
# min_x, max_x, n_samples, noise = -5, 5, 600, 0.5
# x_calib, y_calib, _, _, _ = generate_data_with_heteroscedastic_noise(
#     x_sinx, min_x, max_x, n_samples, noise
# )
#x_calib, y_calib = randomly_generate_cube(300)
@ignore_warnings(category=ConvergenceWarning)
def train_and_run_ci(model,x_train,y_train,x_test,y_test, alpha, function_name,conformal_method, param, real_data):
    #cqr doesnt currently work
    start = time.time()
    if conformal_method == "cqr":
        mapie = MapieQuantileRegressor(estimator, **param)
        #mapie.fit(x_train, y_train,X_calib=x_calib, y_calib=y_calib, random_state=42)
        mapie.fit(x_train, y_train, random_state=42)
        y_pred, y_ci = mapie.predict(x_test, alpha=alpha)

    #All the others work
    else:
        mapie = MapieRegressor(model, **param)
        mapie.fit(x_train, y_train)
        y_pred, y_ci = mapie.predict(x_test, alpha=alpha)
    end = time.time()
    total_time = end-start
    coverage = regression_coverage_score(y_test, y_ci[:,0,0], y_ci[:,1,0])
    mean_width = regression_mean_width_score(y_ci[:,0,0], y_ci[:,1,0])
    #Coverage width based criteration  - designed to both reward narrow intervals and penalize those that do not achieve a specified coverage probability
    #cwc = regression_ssc_score(y_test, y_ci)
    cwc =3

    #marginal coverage rate
   
    #tail coverage rate
   
    if real_data:
        plotting.plot_ci_real_data(
            conformal_method,
            y_test,
            y_pred,
            y_ci[:,0,0],
            y_ci[:,1,0],
            coverage,
            mean_width, function_name)
    else:
        plotting.plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, function_name, conformal_method)    

    return coverage,mean_width, cwc, y_ci, total_time
#%%
#GENERATE DATA

training_data_name = ["sinex_het"] #Function data was generated with
stds = [0,0.3,0.5,0.7,1]
noise_types = ["norm"]


if simulate_data:
    num_samples = 300
    x_train ={}
    y_train = {}
    x_test ={}
    y_test = {}
     
    

    range_start,range_stop,half_norm_loc = -1,1,0.5
    min_x, max_x, n_samples, noise = -5, 5, num_samples, 0.5

    for name in training_data_name:
        match name:
            #sine x with heteroscedastic noise
            case "sinex_het":
                x_train["sinex_het"], y_train["sinex_het"] = gen.generate_data(num_samples, stds,noise_types,"sinex_het")
                x_test["sinex_het"], y_test["sinex_het"] = gen.generate_data(num_samples, stds,noise_types,"sinex_het")
            case "sinex_con":
                x_train["sinex_con"], y_train["sinex_con"] = gen.generate_data(num_samples, stds,noise_types,"sinex_con")
                x_test["sinex_con"], y_test["sinex_con"] = gen.generate_data(num_samples, stds,noise_types,"sinex_con")
            case "linear":
                x_train["linear"], y_train["linear"] = gen.generate_data(num_samples, stds,noise_types)
                x_test["linear"], y_test["linear"] = gen.generate_data(num_samples, stds,noise_types)
            case "nmm":
                x_train["nmm"], y_train["nmm"] = gen.generate_data(num_samples, stds,noise_types,gen.normal_mixture_function)
                x_test["nmm"], y_test["nmm"] = gen.generate_data(num_samples, stds,noise_types,gen.normal_mixture_function)
            case "cube":
                x_train["cube"], y_train["cube"] = gen.generate_data(num_samples, stds,noise_types,"cube")
                x_test["cube"], y_test["cube"] = gen.generate_data(num_samples, stds,noise_types,"cube")
                

            
            
    # x_train["sinex_het"], y_train["sinex_het"], x_test["sinex_het"], y_test["sinex_het"] = gen.generate_data_with_heteroscedastic_noise(
    #     gen.x_sinx, min_x, max_x, n_samples, noise
    # )
   
    # x_train["sinex_con"], y_train["sinex_con"], x_test["sinex_con"], y_test["sinex_con"] = gen.generate_data_with_constant_noise(
    #     gen.x_sinx, min_x, max_x, n_samples, noise
    # )
    # x_train["cube"], y_train["cube"] = gen.randomly_generate_cube(num_samples,range_start,range_stop,half_norm_loc)
    # x_test["cube"], y_test["cube"] = gen.randomly_generate_cube(num_samples,range_start,range_stop,half_norm_loc)
    # #Norm
    # x_train["norm"], y_train["norm"] = gen.generate_data_norm(num_samples)
    # x_test["norm"], y_test["norm"] = gen.generate_data_norm(num_samples)
    # #sine + linear
    # x_train["sine"], y_train["sine"] = gen.generate_data_sin_unif(num_samples)
    # x_test["sine"], y_test["sine"] = gen.generate_data_sin_unif(num_samples)
    # #exponential
    # x_train["exp"], y_train["exp"] = gen.generate_data_exp(num_samples)
    # x_test["exp"], y_test["exp"] = gen.generate_data_exp(num_samples)
#------------------------------------------------------------------------------
#REAL DATA
if real_data:
    file_name = "yacht_hydro.csv"

    df = pd.read_csv(file_name)
    target = "Rr"
    y = df["Rr"]
    columns = df.columns.values.tolist()
    columns.remove(target)
    x = df[columns]
    #x.columns = [None] * len(x.columns)
    #y.columns = [None] * len(y.columns)
    x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)



STRATEGIES = {
    "split" :  dict(method="base",cv='split'),
    "naive": dict(method="naive"),
    # "jackknife": dict(method="base", cv=-1)
    # "jackknife_plus": dict(method="plus"),
    # "jackknife_minmax": dict(method="minmax", cv=-1),
    # "cv": dict(method="base", cv=10),
    # "cv_plus": dict(method="plus", cv=10),
    # "cv_minmax": dict(method="minmax", cv=10),
    # "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50)),
    # "jackknife_minmax_ab": dict(
    #     method="minmax", cv=Subsample(n_resamplings=50)
    # ),
   
    # "cqr": dict(
    #     method="quantile", cv="split", alpha=alpha
    # )
    }

#Slower strategies for debugging
# STRATEGIES = {
#       "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50), conformity_score=GammaConformityScore()),# CHECK THIS WORKS! Gamma conformity score
#     "conformal_quantile_regression": dict(
#         method="quantile", cv="split", alpha=alpha
#     )

#                 }
 
#The MLP
mlp_regressor = MLPRegressor(hidden_layer_sizes=(20,1000), verbose=False, random_state=42)

#Quantile regressor
quantile_model = QuantileRegressor(quantile=alpha, alpha=0, solver=solver)

estimator = LGBMRegressor(
    objective='quantile',
    alpha=alpha,
    random_state=42,
    verbose=-1
)

models = [mlp_regressor]
model_names = ["mlp"] #predictor

#Choose function out of:
"""
sinex_con
sinex_het
exp
cube
norm
sine
"""

if simulate_data:
    for name in training_data_name:
        for std in stds:
            plt.figure()
            moose = x_train[name]
            plt.scatter(x_train[name][std], y_train[name][std], marker="x")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"{name} function example with {num_samples} data points")
            plt.show()
strategy = STRATEGIES.items()

#%%  
coverages = np.array([])
mean_widths = np.array([])
cwcs = np.array([])
results = []
y_cis = {}
for strategy in STRATEGIES.keys():
    y_cis[strategy] = {}
    for func_name in training_data_name:
        y_cis[strategy][func_name] = {}
        
print("Beginning iteration")
#Run through each conformal strategy
if simulate_data:
    for strategy, params in STRATEGIES.items():
        #Run through each data generation technique
        for func_name in training_data_name:
            for std in stds:
                print(f"Performing {strategy} with {func_name} function and {std} std")
                coverage,mean_width, cwc, y_ci, total_time = train_and_run_ci(mlp_regressor,x_train[func_name][std],y_train[func_name][std],x_test[func_name][std],y_test[func_name][std], alpha, func_name,strategy, params, real_data)
                results.append({
                    "strategy": strategy,
                    "data_name": func_name,
                    "coverage": coverage,
                    "mean_width": mean_width,
                    "cwc": cwc,
                    "total_time": total_time,
                    "noise_std":std
                })
                y_cis[strategy][func_name][std] = y_ci
#Plot and calculate real data graphs
elif real_data:
    for strategy, params in STRATEGIES.items():
        print(f"Performing {strategy}")
        coverage,mean_width, cwc, y_ci, total_time = train_and_run_ci(mlp_regressor,x_train,y_train,x_test,y_test, alpha, "yacht data",strategy, params, real_data)
        results.append({
            "strategy": strategy,
            "data_name": file_name,
            "coverage": coverage,
            "mean_width": mean_width,
            "cwc": cwc,
            "total_time": total_time
        })

        #y_cis[strategy] = y_ci
df = pd.DataFrame(results)
df.to_csv(out_file_name, index=False)
#%%

if simulate_data:
    #Plot widths along x
    plotting.plot_widths_along_x(training_data_name, STRATEGIES, x_test,y_cis,stds)
   
    #Plot coverage along x as hist
    num_bins = 11

    for func_name in training_data_name:
        for std in stds:
            heteroscedastic_coverage = plotting.get_coverage(
                y_test, y_cis, STRATEGIES, num_bins, x_test[func_name][std],func_name,std
            )
            
#def plot_coverage_over_std():
    
    
#%%
#Manually set bins if needed
# bins = [0, 1, 2, 3, 4, 5] # make so you don;t have to input the bins
# heteroscedastic_coverage = get_heteroscedastic_coverage(
#     y_test, y_cis, STRATEGIES, bins, x_test, "sinex_het"
# )

# fig = plt.figure()


plotting.coverage_for_each_strategy(out_file_name,stds)
plotting.coverage_for_each_std(out_file_name)
