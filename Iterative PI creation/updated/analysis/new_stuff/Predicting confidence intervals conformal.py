# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:49:01 2024

@author: dan
"""

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
#from lightgbm import LGBMRegressor
#import lightgbm as lgb
# from scikeras.wrappers import KerasRegressor

from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import (regression_coverage_score, regression_mean_width_score, regression_ssc,regression_ssc_score,coverage_width_based,regression_mwi_score)
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
# import tf_to_mapie as tf_map
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point" # for CQR sklearn model


np.random.seed(42)

real_data = 0
multivar_data = 0
simulate_data = 1
if real_data:
    out_file_name = "results_real_data.csv"
    simulate_data = 0
    multivar_data = 0
    num_samples_str = ""
elif multivar_data:
    out_file_name = "linear_results.csv"
    simulate_data = 0
    real_data = 0
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


# =============================================================================
# Prediction and training
# =============================================================================

@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def train_and_run_ci(model,x_train,y_train,x_test,y_test,x_calib,y_calib, alpha, function_name,conformal_method, param, real_data, noise_type,std,num_samples):
    #cqr doesnt currently work

    start = time.time()
    if conformal_method == "cqr":
        mapie = MapieQuantileRegressor(estimator, **param)
        mapie.fit(x_train, y_train,X_calib=x_calib, y_calib=y_calib, random_state=42)
        #mapie.fit(x_train, y_train.ravel(), random_state=42)
        y_pred, y_ci = mapie.predict(x_test, alpha=alpha)

    #All the others work
    else:
        mapie = MapieRegressor(model, **param, verbose=0)
        mapie.fit(x_train, y_train.ravel())
        y_pred, y_ci = mapie.predict(x_test, alpha=alpha)
    end = time.time()
    total_time = end-start
   
    coverage = regression_coverage_score(y_test, y_ci[:,0,0], y_ci[:,1,0])
    mean_width = regression_mean_width_score(y_ci[:,0,0], y_ci[:,1,0])
    max_width = max(y_ci[:,1,0]-y_ci[:,0,0])

    ref_length = np.subtract(
        float(y_test.max()),
        float(y_test.min())
    )
    norm_mean_width = mean_width/ref_length
    print(norm_mean_width)
    try:
        ssc = regression_ssc(y_test, y_ci, num_bins=5)
    except:
        ssc = 999    
    cwc = coverage_width_based(y_test,
        y_ci[:, 0, 0],
        y_ci[:, 1, 0],
        eta=0.001, #param that balances the contributions of the MPIW and coverage
        alpha=alpha) #Coverage width based criteration  - designed to both reward narrow intervals and penalize those that do not achieve a specified coverage probability
    mwi = regression_mwi_score(y_test,y_ci,alpha)
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
            mean_width, function_name, layer_size_str)
        plotting.plot_coverage_along_quantiles(conformal_method,y_test,y_pred,y_ci[:,0,0],y_ci[:,1,0],x_test,function_name, conformal_method, layer_size_str)
       
    elif simulate_data:
        plotting.plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, function_name, conformal_method, noise_type, std,num_samples, layer_size_str)    
        # cov_along_x = plotting.get_coverage(
        #         y_test, y_cis, STRATEGIES, num_bins, x_test[func_name][std],func_name,std, num_samples_str, noise_type, layer_size_str
        #     )
   
    #save to results/STRATEGY/std
    return coverage,mean_width, cwc,mwi, y_ci, total_time,ssc, norm_mean_width
#%%

plot_data_gen_funcs = 0

#GENERATE DATA
out_file_name = "tester_coverage.csv" #REMEBER TO CHANGE THE FILENAME

layer_size_array = [(200, 1000)]
#layer_size_array = [(200, 1000)]
#num_samples_array = [300,500,1000] #Training data size
num_samples_array = [1000]
#cv_sizes = [5,10,20,50]
cv_sizes = [50]
#training_data_names = [["sinex_het"],["sinex_con"],["linear"],["cube"],["nmm"]] #Function data is generated with - iterates over this
#training_data_names = [["linear_multi"],["poly_multi"],["non_linear_multi"]]
training_data_names = [["nmm"]]
#noise_types = ["normal","block","student","laplace"]
noise_types = ["normal"]
dataset_names = ["nothing"]
optimisers = ["adam"]
#optimisers = ["adam"]
#bagging_fractions = [0.1,0.3,0.5,1] # already done "1" with cv=50
bagging_fractions = [1]
num_features_arr = [3,5,10]
active_functions = ['relu'] #
if real_data:
    #out_file_name = "real_results_boston_yacht_hour_concrete.csv"
    out_file_name = "real_results_boston_yacht_hour_concrete.csv"
    training_data_names = [["linear"]]
    noise_types = ["normal"]
    num_samples_array = [300]
    layer_size_array = [(400, 2000, 2000, 2000)]
    #dataset_names = ["yacht_hydro.csv", "boston.csv","hour.csv", "concrete.csv"] #hour is the bike UCI dataset
    dataset_names = ["boston.csv","hour.csv", "concrete.csv"]
    cv_sizes = [5]
    bagging_fractions = [1]
    
current_iter = 1
for num_samples in num_samples_array:
    for layer_size in layer_size_array:
        for training_data_name in training_data_names:
            for noise_type in noise_types:
                for dataset_name in dataset_names:
                    for cv_size in cv_sizes:
                        for optimiser in optimisers:
                            for bagging_frac in bagging_fractions:
                                for num_features in num_features_arr:
                                    for active_function in active_functions:
                                     
                                        total_iterations = len(num_samples_array)*len(layer_size_array)*len(training_data_names)*len(noise_types)*len(dataset_names)*len(cv_sizes)*len(optimisers)*len(bagging_fractions)*len(num_features_arr)*len(active_functions)
                                        print(current_iter, "/",total_iterations)
                                        print(f"{num_samples}_{layer_size}_{training_data_name}_{noise_type}")
                                        layer_size_str=  str(layer_size)
                                        match noise_type:
                                            case "normal":
                                                #stds = [0, 0.2,0.5,1] #normal
                                                stds = [0.5]
                                            case "block":
                                                stds = [0.01,0.2,0.5,1] #block normal
                                            case "student":
                                                stds = [4,5,10,1000] #student t - wild noise
                                            case "cauchy":
                                                stds = [0.001,0.005,0.008,0.01] #Cauchy std - VERY WILD NOISE - MAYBE USE FOR OUTLIER DETECTION
                                            case "laplace":
                                                stds = [0.5,1,1.5,2] #Laplace - more predictable noise
                                            case _:
                                                stds = [100000] #failure
                       
                                        if simulate_data or multivar_data:
                                            test_set_size = 300
                                            num_samples_str = str(num_samples)
                                            x_train ={}
                                            y_train = {}
                                            x_test ={}
                                            y_test = {}
                                            x_calib ={}
                                            y_calib = {}
                                             
                                            range_start,range_stop,half_norm_loc = -1,1,0.5
                                            min_x, max_x, n_samples, noise = -5, 5, num_samples, 0.5
                                       
                                            for name in training_data_name:
                                                match name:
                                                    #sine x with heteroscedastic noise num_samples=300,stds=[0.3,0.5,0.7,1],noise_type="norm", function=linear_function
                                                    case "sinex_het":
                                                        x_train["sinex_het"], y_train["sinex_het"] = gen.generate_data(num_samples, stds,noise_type,"sinex_het")
                                                        x_test["sinex_het"], y_test["sinex_het"] = gen.generate_data(test_set_size, stds,noise_type,"sinex_het")
                                                        x_calib["sinex_het"], y_calib["sinex_het"] = gen.generate_data(num_samples, stds,noise_type,"sinex_het")
                                                    case "sinex_con":
                                                        x_train["sinex_con"], y_train["sinex_con"] = gen.generate_data(num_samples, stds,noise_type,"sinex_con")
                                                        x_test["sinex_con"], y_test["sinex_con"] = gen.generate_data(test_set_size, stds,noise_type,"sinex_con")
                                                        x_calib["sinex_con"], y_calib["sinex_con"] = gen.generate_data(num_samples, stds,noise_type,"sinex_con")
                                                    case "linear":
                                                        x_train["linear"], y_train["linear"] = gen.generate_data(num_samples, stds,noise_type)
                                                        x_test["linear"], y_test["linear"] = gen.generate_data(test_set_size, stds,noise_type)
                                                        x_calib["linear"], y_calib["linear"] = gen.generate_data(num_samples, stds,noise_type)
                                                    case "nmm":
                                                        #normal mixture model
                                                        x_train["nmm"], y_train["nmm"] = gen.generate_data(num_samples, stds,noise_type,gen.normal_mixture_function)
                                                        x_test["nmm"], y_test["nmm"] = gen.generate_data(test_set_size, stds,noise_type,gen.normal_mixture_function)
                                                        x_calib["nmm"], y_calib["nmm"] = gen.generate_data(num_samples, stds,noise_type,gen.normal_mixture_function)
                                                    case "cube":
                                                        x_train["cube"], y_train["cube"] = gen.generate_data(num_samples, stds,noise_type,"cube")
                                                        x_test["cube"], y_test["cube"] = gen.generate_data(test_set_size, stds,noise_type,"cube")
                                                        x_calib["cube"], y_calib["cube"] = gen.generate_data(num_samples, stds,noise_type,"cube")
                                                    case "linear_multi":
                                                        x_train["linear_multi"], y_train["linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "linear_multi")
                                                        x_test["linear_multi"], y_test["linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "linear_multi")
                                                        x_calib["linear_multi"], y_calib["linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "linear_multi")
                                                    case "poly_multi":
                                                        x_train["poly_multi"], y_train["poly_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "poly_multi")
                                                        x_test["poly_multi"], y_test["poly_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "poly_multi")
                                                        x_calib["poly_multi"], y_calib["poly_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "poly_multi")
                                                    case "non_linear_multi":
                                                        x_train["non_linear_multi"], y_train["non_linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "non_linear_multi")
                                                        x_test["non_linear_multi"], y_test["non_linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "non_linear_multi")
                                                        x_calib["non_linear_multi"], y_calib["non_linear_multi"] = gen.generate_data_multi(num_samples,num_features, stds,noise_types, "non_linear_multi")
                                               
                                        #------------------------------------------------------------------------------
                                        #REAL DATA
                                        if real_data:
                                            file_prefix = "real_datasets"
                                            file_name = os.path.join(file_prefix, dataset_name)
                                       
                                            df = pd.read_csv(file_name)
                                            match dataset_name:
                                                case "hour.csv":
                                                    target = "cnt"
                                                    df['day'] = df['dteday'].apply(lambda x: int(x.split('-')[0]))
                                                    df['month'] = df['dteday'].apply(lambda x: int(x.split('-')[1]))
                                                   
                                                    # Drop the 'dteday' column
                                                    df = df.drop('dteday', axis=1)
                                                case "yacht_hydro.csv":
                                                    target = "Rr"
                                                case "boston.csv":
                                                    target = "MEDV"
                                                case "concrete.csv":
                                                    target = "ccs"
                                                    
                                            y = df[target]
                                            columns = df.columns.values.tolist()
                                            columns.remove(target)
                                            x = df[columns]
                                            # try:
                                            #     x["LC"] = x["LC"].abs()
                                            # except:
                                            #     pass
                                            x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)
                                            x_calib = 0
                                            y_calib = 0
                                            dataset_name_str  = os.path.splitext(os.path.basename(dataset_name))[0]
                                        if real_data:
                                            n_samples = len(x_train)
                                        n_samples_boostrap = int(bagging_frac*n_samples)
                                       
                                        STRATEGIES = {
                                            "split" :  dict(method="base",cv='split'),
                                            "split_resid_norm" :  dict(method="base",cv='split', conformity_score=ResidualNormalisedScore()),
                                            "split_gamma" :  dict(method="base",cv='split', conformity_score=GammaConformityScore()),
                                            # "naive": dict(method="naive"),
                                            # "jackknife": dict(method="base", cv=-1)
                                            # "jackknife_plus": dict(method="plus", cv=-1),
                                            # "jackknife_minmax": dict(method="minmax", cv=-1),
                                            # "cv": dict(method="base", cv=cv_size),
                                            # "cv_plus": dict(method="plus", cv=cv_size),
                                            # "cv_minmax": dict(method="minmax", cv=10),
                                            # "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=cv_size, n_samples=n_samples_boostrap)),
                                            # "jackknife_minmax_ab": dict(
                                            #     method="minmax", cv=Subsample(n_resamplings=50)
                                            # ),
                                           
                                            # "cqr": dict(
                                            #     method="quantile", cv="split", alpha=alpha
                                            # )
                                            }
                                         
                                        #The MLP
                                        mlp_regressor = MLPRegressor(hidden_layer_sizes=layer_size, verbose=False, random_state=42, early_stopping=True, solver=optimiser, activation=active_function, max_iter=20) #200,1000
                                        # mlp_regressor = tf_map.get_model((num_features,),active_function,optimiser, layer_size)
                                        
                                        # mlp_regressor = tf_map.TensorflowToMapie(mlp_regressor)
                                                                           
                                        #Quantile regressor
                                        # quantile_model = QuantileRegressor(quantile=alpha, alpha=0, solver=solver)
                                       
                                        # estimator = LGBMRegressor(
                                        #     objective='quantile',
                                        #     alpha=alpha,
                                        #     random_state=42,
                                        #     verbose=-1
                                        # )
                                       
                                        if plot_data_gen_funcs:
                                            if simulate_data:
                                                for name in training_data_name:
                                                    for std in stds:
                                                       
                                                        plt.figure()
                                                        moose = x_train[name]
                                                        plt.scatter(x_train[name][std], y_train[name][std], marker="x")
                                                        #plt.scatter(x_test[name][std], y_test[name][std], marker="x")
                                                        plt.xlabel("x")
                                                        plt.ylabel("y")
                                                        plt.title(f"{name} function example with {num_samples} data points and {noise_type} noise")
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
                                            if real_data:
                                                existing_df = pd.DataFrame(columns=["datset","strategy", "data_name", "coverage", "mean_width", "cwc", "mwi","total_time","layer_size","ssc","cv_size", "optimiser", "bagging_fraction", "normalised_mean_width" ])
                                            else:
                                               
                                                existing_df = pd.DataFrame(columns=["strategy", "data_name", "coverage", "mean_width", "cwc", "mwi","total_time","noise_std","num_samples","noise_type","layer_size","ssc","cv_size", "optimiser", "bagging_fraction", "normalised_mean_width", "num_features","active_function"])
                                       
                                        print("Beginning iteration")
                                        #Run through each conformal strategy
                                        if simulate_data or multivar_data:
                                            orig_cv_size = cv_size
                                            for strategy, params in STRATEGIES.items():
                                                #Run through each data generation technique
                                                for func_name in training_data_name:
                                                    for std in stds:
                                                        print(f"Performing {strategy} with {func_name} function and {std} std")
                                                        coverage,mean_width, cwc,mwi, y_ci, total_time, ssc,norm_mean_width = train_and_run_ci(
                                                            mlp_regressor,x_train[func_name][std],y_train[func_name][std],x_test[func_name][std],y_test[func_name][std], x_calib[func_name][std],y_calib[func_name][std],  
                                                            alpha, func_name,strategy, params, real_data, noise_type, std,num_samples_str
                                                            )
                                                        if (strategy == "jackknife_plus") or (strategy == "jackknife"):
                                                            cv_size = num_samples
                                                        else:
                                                            cv_size = orig_cv_size
                                                        results.append({
                                                            "strategy": strategy,
                                                            "data_name": func_name,
                                                            "coverage": coverage,
                                                            "mean_width": mean_width,
                                                            "cwc": cwc, #coverage width based criterion
                                                            "mwi": mwi, #mean winkler score
                                                            "total_time": total_time,
                                                            "noise_std":std,
                                                            "num_samples": num_samples,
                                                            "noise_type": noise_type,
                                                            "layer_size": layer_size_str,
                                                            "ssc": ssc,
                                                            "cv_size": cv_size,
                                                            "optimiser" : optimiser,
                                                            "bagging_fraction" : bagging_frac,
                                                            "normalised_mean_width" : norm_mean_width,
                                                            "num_features" : num_features,
                                                            "active_function" : active_function
                                                        })
                                                        y_cis[strategy][func_name][std] = y_ci
                                        #Plot and calculate real data graphs
                                        elif real_data:
                                            for strategy, params in STRATEGIES.items():
                                                print(f"Performing {strategy}")
                                                coverage,mean_width, cwc,mwi, y_ci, total_time,ssc,norm_mean_width = train_and_run_ci(
                                                    mlp_regressor,x_train,y_train,x_test,y_test,x_calib,y_calib, alpha, dataset_name_str,strategy, params, real_data, noise_type, 0,"string"
                                                    )
                                                results.append({
                                                    "dataset" : dataset_name_str,
                                                    "strategy": strategy,
                                                    "data_name": file_name,
                                                    "coverage": coverage,
                                                    "mean_width": mean_width,
                                                    "cwc": cwc,
                                                    "mwi": mwi,
                                                    "total_time": total_time,
                                                    "layer_size": layer_size_str,
                                                    "ssc": ssc,
                                                    "cv_size" : cv_size,
                                                    "optimiser" : optimiser,
                                                    "bagging_fraction" : bagging_frac,
                                                    "normalised_mean_width" : norm_mean_width
                                                })
                                       
                                                #y_cis[strategy] = y_ci
                                               
                                               
                                        print("============================================================================")
                                        #%%
                                        #convert results to a DataFrame
                                        new_df = pd.DataFrame(results)
                                       
                                        #append the new data to the existing data
                                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                                        #%%
                                        #save the combined data back to the CSV file
                                        combined_df.to_csv(out_file_name, index=False)
                                        #%%
                                       
                                        want_coverage_along_x = 1
                                        plot_coverage_width_strategies = 1
                                        plot_coverage_width_stds = 0
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
                                                            y_test, y_cis, STRATEGIES, num_bins, x_test[func_name][std],func_name,std, num_samples_str, noise_type, layer_size_str
                                                        )
                                           
                                        #%%
                                       
                                        if plot_coverage_width_strategies:
                                            plotting.coverage_for_each_strategy(out_file_name,stds, noise_type,num_samples_str, layer_size_str)
                                        if plot_coverage_width_stds:
                                            plotting.coverage_for_each_std(out_file_name, noise_type,num_samples_str, layer_size_str)
       
                                        current_iter += 1
#%%
# base_path = 'results'
# function_name = "linear"
# noise_type = "laplace"
# strategy = "jackknife_plus_ab"
# match noise_type:
#     case "normal":
#         noise_parameter_name = "STD"
#         plot_stds = [0,0.2,0.5,1] #normal
#     case "block":
#         noise_parameter_name = "Spike STD"
#         plot_stds = [0.01,0.2,0.5,1] #block normal
#     case "student":
#         noise_parameter_name = "Scale Parameter"
#         plot_stds = [4,5,10,1000] #student t - wild noise
#     case "cauchy":
#         plot_stds = [0.001,0.005,0.008,0.01] #Cauchy std - VERY WILD NOISE - MAYBE USE FOR OUTLIER DETECTION
#         noise_parameter_name = "Scale Parameter"
#     case "laplace":
#         plot_stds = [0.5,1,1.5,2] #Laplace - more predictable noise
#         noise_parameter_name = "Scale Parameter"
#     case _:
#         noise_parameter_name = "Scale Parameter"
# plotter = plotting.Plotter(base_path)
# plotter.plot_images(function_name,num_samples_str, noise_type, plot_stds, strategy, noise_parameter_name, layer_size_str)