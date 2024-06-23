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
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
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
import os

#import relevant code
import generate_samples as gen
import plotting_functions as plotting
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point" # for CQR sklearn model


np.random.seed(42)
gamma_score = 0
real_data = 0
if real_data:
    out_file_name = "results_real_data.csv"
    simulate_data = 0
else:
    out_file_name = "results.csv"
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
# def build_tf_mlp(hidden_layer_sizes=(200, 500), input_shape=(None,), random_state=42):
#     # Set the random seed for reproducibility
#     tf.random.set_seed(random_state)
    
#     # Initialize the Sequential model
#     model = Sequential()
    
#     # Add the input layer
#     model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=input_shape))
    
#     # Add hidden layers
#     for layer_size in hidden_layer_sizes[1:]:
#         model.add(Dense(layer_size, activation='relu'))
    
#     # Add the output layer
#     model.add(Dense(1))  # Assuming a single output for regression
    
#     # Compile the model
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
#     return model

# @ignore_warnings(category=ConvergenceWarning)
# @ignore_warnings(category=UserWarning)
def train_and_run_ci(model,x_train,y_train,x_test,y_test, alpha, function_name,conformal_method, param, real_data, noise_type,std):
    #cqr doesnt currently work
    start = time.time()
    if conformal_method == "cqr":
        mapie = MapieQuantileRegressor(estimator, **param)
        #mapie.fit(x_train, y_train,X_calib=x_calib, y_calib=y_calib, random_state=42)
        mapie.fit(x_train, y_train.ravel(), random_state=42)
        y_pred, y_ci = mapie.predict(x_test, alpha=alpha)

    #All the others work
    else:
        mapie = MapieRegressor(model, **param)
        mapie.fit(x_train, y_train.ravel())
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
        plotting.plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, function_name, conformal_method, noise_type, std)    
    
    #save to results/STRATEGY/std
    return coverage,mean_width, cwc, y_ci, total_time
#%%
#GENERATE DATA

training_data_name = ["sinex_het"] #Function data is generated with - iterates over this
stds = [0,0.3,0.4,0.5,1,2] # Iterates over the noise standard deviation 
noise_type = "block"


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
                x_train["sinex_het"], y_train["sinex_het"] = gen.generate_data(num_samples, stds,noise_type,"sinex_het")
                x_test["sinex_het"], y_test["sinex_het"] = gen.generate_data(num_samples, stds,noise_type,"sinex_het")
            case "sinex_con":
                x_train["sinex_con"], y_train["sinex_con"] = gen.generate_data(num_samples, stds,noise_type,"sinex_con")
                x_test["sinex_con"], y_test["sinex_con"] = gen.generate_data(num_samples, stds,noise_type,"sinex_con")
            case "linear":
                x_train["linear"], y_train["linear"] = gen.generate_data(num_samples, stds,noise_type)
                x_test["linear"], y_test["linear"] = gen.generate_data(num_samples, stds,noise_type)
            case "nmm":
                #normal mixture model
                x_train["nmm"], y_train["nmm"] = gen.generate_data(num_samples, stds,noise_type,gen.normal_mixture_function)
                x_test["nmm"], y_test["nmm"] = gen.generate_data(num_samples, stds,noise_type,gen.normal_mixture_function)
            case "cube":
                x_train["cube"], y_train["cube"] = gen.generate_data(num_samples, stds,noise_type,"cube")
                x_test["cube"], y_test["cube"] = gen.generate_data(num_samples, stds,noise_type,"cube")
                
        
            
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
    "split_resid_norm" :  dict(method="base",cv='split', conformity_score=ResidualNormalisedScore()),
    # "naive": dict(method="naive"),
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
mlp_regressor = MLPRegressor(hidden_layer_sizes=(200,500,), verbose=False, random_state=42, early_stopping=True, solver="adam") #20,1000
#input_shape = (num_samples,)
#mlp_regressor = build_tf_mlp(hidden_layer_sizes=(200, 500), input_shape=input_shape)

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
            plt.scatter(x_test[name][std], y_test[name][std], marker="x")
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
        

#check if the file already exists
if os.path.exists(out_file_name):
    # Load existing data
    existing_df = pd.read_csv(out_file_name)
else:
    #create an empty DataFrame with the same structure
    existing_df = pd.DataFrame(columns=["strategy", "data_name", "coverage", "mean_width", "cwc", "total_time","noise_std","num_samples","noise_type"])

print("Beginning iteration")
#Run through each conformal strategy
if simulate_data:
    for strategy, params in STRATEGIES.items():
        #Run through each data generation technique
        for func_name in training_data_name:
            for std in stds:
                print(f"Performing {strategy} with {func_name} function and {std} std")
                coverage,mean_width, cwc, y_ci, total_time = train_and_run_ci(
                    mlp_regressor,x_train[func_name][std],y_train[func_name][std],x_test[func_name][std],y_test[func_name][std],
                    alpha, func_name,strategy, params, real_data, noise_type, std
                    )
                results.append({
                    "strategy": strategy,
                    "data_name": func_name,
                    "coverage": coverage,
                    "mean_width": mean_width,
                    "cwc": cwc,
                    "total_time": total_time,
                    "noise_std":std,
                    "num_samples": num_samples,
                    "noise_type": noise_type
                })
                y_cis[strategy][func_name][std] = y_ci
#Plot and calculate real data graphs
elif real_data:
    for strategy, params in STRATEGIES.items():
        print(f"Performing {strategy}")
        coverage,mean_width, cwc, y_ci, total_time = train_and_run_ci(
            mlp_regressor,x_train,y_train,x_test,y_test, alpha, "yacht data",strategy, params, real_data, noise_type, std
            )
        results.append({
            "strategy": strategy,
            "data_name": file_name,
            "coverage": coverage,
            "mean_width": mean_width,
            "cwc": cwc,
            "total_time": total_time
        })

        #y_cis[strategy] = y_ci
        
        
print("============================================================================")
#convert results to a DataFrame
new_df = pd.DataFrame(results)

#append the new data to the existing data
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

#save the combined data back to the CSV file
combined_df.to_csv(out_file_name, index=False)
#%%

want_coverage_along_x = 0
plot_coverage_width_strategies = 1
plot_coverage_width_stds = 1
want_widths_along_x = 0

if simulate_data:
    #Plot widths along x
    if want_widths_along_x:
        plotting.plot_widths_along_x(training_data_name, STRATEGIES, x_test,y_cis,stds)
   
    #Plot coverage along x as hist
    num_bins = 11
    if want_coverage_along_x:
        for func_name in training_data_name:
            for std in stds:
                heteroscedastic_coverage = plotting.get_coverage(
                    y_test, y_cis, STRATEGIES, num_bins, x_test[func_name][std],func_name,std
                )
            
    
#%%

if plot_coverage_width_strategies:
    plotting.coverage_for_each_strategy(out_file_name,stds, noise_type)
if plot_coverage_width_stds:
    plotting.coverage_for_each_std(out_file_name, noise_type)
    
    
#%%
base_path = 'results'
function_name = "sinex_het"
noise_type = "block"
strategy = "split"

plotter = plotting.Plotter(base_path)
plotter.plot_images(function_name, noise_type, stds, strategy)
