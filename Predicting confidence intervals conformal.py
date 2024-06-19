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


solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point" # for CQR sklearn model


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
# =============================================================================
# Simulate data functions
# =============================================================================
#Cubic
def randomly_generate_cube(num_samples,range_start=-1,range_stop=1,half_norm_loc = 1,  std_dev=0.4):
    """
    y = x**3 + normal noise
    9 in 10 chance for x to be x= uniform(-1,1), if not, normally distributed with mean 1 and std 0.55, means some values are very sparse and far away
    from https://proceedings.neurips.cc/paper/2021/file/46c7cb50b373877fb2f8d5c4517bb969-Paper.pdf
    """
    y_train = []
    x_train = []
    mean = 1
    standard_deviation=0.55 #'1' was ok ish as well
    for i in range(num_samples):
       
        decider = random.randrange(1,10) #Produces a integer between the range
       
        if decider <=8:
            x = random.uniform(range_start, range_stop) #Produces a random float value
        else:
            x = halfnorm.rvs(loc = half_norm_loc, scale = standard_deviation, size=1)
            x=x[0]
            #x = np.random.normal(loc=1, scale=standard_deviation)
        noise = np.random.normal(loc=0, scale=std_dev)
        y = x**3 + noise 
        y_train.append(y)  # Append standard_deviation as a new row
        x_train.append(x)  # Append gaussian vector as a new row
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #plt.figure()
    #plt.scatter(x_train,y_train)
    # Sort x_train and ensure y_train matches the sorted x_train
    sorted_indices = np.argsort(x_train)
    x_train_sorted = x_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]
    x_train_sorted = np.transpose(x_train_sorted)
    y_train_sorted = np.transpose(y_train_sorted)
    return  x_train_sorted.reshape(-1,1), y_train_sorted.reshape(-1,1)

#%%
#Normally distributed but hetorscedastic - very tricky for mlp to solve -  easy for CQR
def generate_data_norm(num_samples, std_dev=0.5):
    x_train = np.linspace(start=0, stop=10, num=num_samples)
    y_true_mean = 10 + 0.5 * x_train**3
    y_train = y_true_mean + rng.normal(loc=0, scale=std_dev + 0.5 * x_train**3, size=x_train.shape[0])
    #plt.scatter(x_train,y_train)
    return np.transpose(np.array([x_train])), np.transpose(np.array([y_train])) #y_true_mean

#Sine function + linear
def generate_data_sin_unif(num_samples, std_dev=0.7):
    """
    y = 2*sin(pi*x ) + pi*x + noise
    Noise is normal
    From Adaptive, distribution-free prediction intervals for deep networks
    """
    def f(Z):
        return(2.0*np.sin(np.pi*Z) + np.pi*Z)
    p = 1
    X = np.random.uniform(size=(num_samples,p))
    beta = np.zeros((p,))
    beta[0:5] = 1.0
    Z = np.dot(X,beta)
    E = np.random.normal(size=(num_samples,),scale=std_dev)
    #Y = f(Z) + np.sqrt(1.0+Z**2) * E
    Y = f(Z) + E
    return X.astype(np.float32), np.transpose(np.array([Y.astype(np.float32)]))

def generate_data_exp(num_samples, std_dev=2):
    "Uniformly distribute x points between -1 and 3. f(x) is a y = e^x + noise. Noise is normal with SD 2"
    y_train = []
    x_train = []
    mean = 1
    standard_deviation=1
   
       
    rv = norm(loc = mean, scale = standard_deviation)
    x = np.linspace(start=-1, stop=3, num=num_samples)
    #x = np.random.normal(loc=mean, scale=standard_deviation)
    noise = np.random.normal(loc=1, scale=std_dev,size=(num_samples,))
    y = np.exp(x) + noise
    y_train.append(y)  # Append standard_deviation as a new row
    x_train.append(x)  # Append gaussian vector as a new row
       
    return np.transpose(np.array(x_train)), np.transpose(np.array(y_train))

def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x*np.sin(x)

def generate_data_with_heteroscedastic_noise(funct, min_x, max_x, n_samples, noise):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples*5)
    y_train = (
        funct(X_train) +
        (np.random.normal(0, noise, len(X_train)) * X_train)
    )
    y_test = (
        funct(X_test) +
        (np.random.normal(0, noise, len(X_test)) * X_test)
    )
    y_mesh = funct(X_test)
    return (
        X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh
    )



def generate_data_with_constant_noise(funct, min_x, max_x, n_samples, noise):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples*5)
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += np.random.normal(0, noise, y_train.shape[0])
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return (
        X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh
    )


#%%
# =============================================================================
# Plotting
# =============================================================================

def plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, name, conformal_method):
    # for val in range(len(y_ci[:, 0, 0])):
    #     print(y_ci[val, 0, 0]-y_ci[val, 1, 0])
    order = np.argsort(x_test.flatten())#set order for coordinates
    plt.figure(figsize=(8, 8))
    plt.plot(
        x_test[order],
        y_pred[order],
        label="Predictions",
        color="green"
    )
    plt.fill_between(
        x_test.flatten()[order],
        y_ci[:, 0, 0][order],
        y_ci[:, 1, 0][order],
        alpha=0.4,
        label="Confidence Intervals",
        color="green"
    )
    plt.scatter(x_test, y_test, color="red", alpha=0.7, label="testing", s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{conformal_method} confidence intervals for MLP with {name} function with total coverage {coverage:.2f} and mean width {mean_width:.2f}")
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
        ncol=3
    )
    plt.show()

#Returns data coverage within specific x intervals for plotting
def get_coverage(y_test, y_cis, STRATEGIES, num_bins, x_test, func_name):
    recap = {}
    bins = np.linspace(min(x_test),max(x_test),num_bins)
    bin_labels = []
    for i in range(len(bins)-1):
        bin1, bin2 = bins[i], bins[i+1]
        bin1lab, bin2lab = float(bins[i]), float(bins[i + 1])
        bin_labels.append(f"[{bin1lab:.2f}, {bin2lab:.2f}]")
        name = f"[{bin1}, {bin2}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where((x_test >= bins[i]) * (x_test < bins[i+1]))
            y_test_trunc = np.take(y_test[func_name], indices)
            y_low_ = np.take(y_cis[strategy][func_name][:, 0, 0], indices)
            y_high_ = np.take(y_cis[strategy][func_name][:, 1, 0], indices)
            score_coverage = regression_coverage_score(
                y_test_trunc[0], y_low_[0], y_high_[0]
            )
            recap[name].append(score_coverage)
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    recap_df.T.plot.bar(figsize=(12, 5), alpha=0.7)
    plt.axhline(1-alpha, ls="--", color="k")
    plt.ylabel("Conditional Coverage")
    plt.xlabel("X bins")
    plt.xticks(ticks=np.arange(len(bin_labels)), labels=bin_labels, rotation=45, ha='right')
    plt.ylim(0.6, 1.0)
    plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title(f"Coverage for each conformal strategy with {func_name} noise function")
    plt.show()
    return recap_df


def get_heteroscedastic_coverage(y_test, y_cis, STRATEGIES, bins, x_test, func_name):
    recap = {}
    for i in range(len(bins)-1):
        bin1, bin2 = bins[i], bins[i+1]
        name = f"[{bin1}, {bin2}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where((x_test[func_name] >= bins[i]) * (x_test[func_name] <= bins[i+1]))
            y_test_trunc = np.take(y_test[func_name], indices)
            y_low_ = np.take(y_cis[strategy][func_name][:, 0, 0], indices)
            y_high_ = np.take(y_cis[strategy][func_name][:, 1, 0], indices)
            score_coverage = regression_coverage_score(
                y_test_trunc[0], y_low_[0], y_high_[0]
            )
            recap[name].append(score_coverage)
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    recap_df.T.plot.bar(figsize=(12, 5), alpha=0.7)
    plt.axhline(1-alpha, ls="--", color="k")
    plt.ylabel("Conditional coverage")
    plt.xlabel("x bins")
    plt.xticks(rotation=0)
    plt.ylim(0.8, 1.0)
    plt.legend(fontsize=8, loc="upper right")
    plt.title("Coverage for each conformal strategy with heteroscedastic noise function")
    plt.show()
    return recap_df

def plot_widths_along_x(training_data_name, STRATEGIES, x_test,y_cis):

    for name in training_data_name:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for strategy in STRATEGIES.keys():
            ax.plot(
                x_test[name],
                y_cis[strategy][name][:, 1, 0].ravel() - y_cis[strategy][name][:, 0, 0].ravel(),
                label=strategy
            )
        ax.set_xlabel("x")
        ax.set_ylabel("Confidence Interval Width")
        ax.legend(fontsize=8)
        ax.set_title(f"Confidence interval widths for {name} function")
        plt.show()
       
def plot_ci_real_data(method,y_test_sorted,y_pred_sorted,lower_bound,upper_bound,coverage,width, title):

    lower_bound_ = lower_bound
    y_pred_sorted_ = y_pred_sorted
    y_test_sorted_ = y_test_sorted
   
    # lower_bound_ = lower_bound_[0:np.round(len(lower_bound_)/3)]
    # y_pred_sorted_ = y_pred_sorted_[0:np.round(len(y_pred_sorted_)/3)]
    # y_test_sorted_ = y_test_sorted_[0:np.round(len(y_test_sorted_)/3)]
   
    error = y_pred_sorted_-lower_bound_
    fig, axs = plt.subplots(1,1)
    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=np.abs(error[~warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        label="Inside prediction interval"
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=np.abs(error[warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red",
        label="Outside prediction interval"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*", color="green",
        label="True value"
    )
    axs.set_xlabel("True target value")
    axs.set_ylabel("Predicted target value")
    # ab = AnnotationBbox(
    #     TextArea(
    #         f"Coverage: {coverage:.2f}\n"
    #         + f"Interval width: {width:.2f}"
    #     ),
    #     xy=(0, 1),
    #     )
    # lims = [
    #     np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
    #     np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    # ]
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    min_val = np.min([xlim, ylim])
    max_val = np.max([xlim, ylim])
    x_vals = np.linspace(max_val, 0, 500)
    y_vals = x_vals
    axs.plot(x_vals, y_vals, '--', alpha=0.75, color="black", label="x=y")
    # axs.add_artist(ab)
    axs.set_title(f"Predicted values using the {method} method on {title}. Coverage {coverage:.2f}, MPIW {width:.2f}", fontweight='bold')
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
        plot_ci_real_data(
            conformal_method,
            y_test,
            y_pred,
            y_ci[:,0,0],
            y_ci[:,1,0],
            coverage,
            mean_width, function_name)
    else:
        plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, function_name, conformal_method)    

    return coverage,mean_width, cwc, y_ci, total_time
#%%
#GENERATE DATA
if simulate_data:
    num_samples = 300
    x_train ={}
    y_train = {}
    x_test ={}
    y_test = {}
     
    #sine x with heteroscedastic noise - y_mesh is the true function without noise
    if gamma_score:
        range_start,range_stop,half_norm_loc = -1,1,0.5
        min_x, max_x, n_samples, noise = -5, 5, num_samples, 0.5
    else:
        range_start,range_stop,half_norm_loc = -1,1,1
        min_x, max_x, n_samples, noise = -5, 5, num_samples, 0.5
    x_train["sinex_het"], y_train["sinex_het"], x_test["sinex_het"], y_test["sinex_het"], y_mesh = generate_data_with_heteroscedastic_noise(
        x_sinx, min_x, max_x, n_samples, noise
    )
   
    x_train["sinex_con"], y_train["sinex_con"], x_test["sinex_con"], y_test["sinex_con"], y_mesh = generate_data_with_constant_noise(
        x_sinx, min_x, max_x, n_samples, noise
    )
    x_train["cube"], y_train["cube"] = randomly_generate_cube(num_samples,range_start,range_stop,half_norm_loc)
    x_test["cube"], y_test["cube"] = randomly_generate_cube(num_samples,range_start,range_stop,half_norm_loc)
    #Norm
    x_train["norm"], y_train["norm"] = generate_data_norm(num_samples)
    x_test["norm"], y_test["norm"] = generate_data_norm(num_samples)
    #sine + linear
    x_train["sine"], y_train["sine"] = generate_data_sin_unif(num_samples)
    x_test["sine"], y_test["sine"] = generate_data_sin_unif(num_samples)
    #exponential
    x_train["exp"], y_train["exp"] = generate_data_exp(num_samples)
    x_test["exp"], y_test["exp"] = generate_data_exp(num_samples)
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
    # "split" :  dict(method="base",cv='split',conformity_score=ResidualNormalisedScore()),
    # "naive": dict(method="naive",conformity_score=GammaConformityScore()),
    # "jackknife": dict(method="base", cv=-1)
    # "jackknife_plus": dict(method="plus"),
    # # # "jackknife_minmax": dict(method="minmax", cv=-1),
    # # # "cv": dict(method="base", cv=10),
    # # # "cv_plus": dict(method="plus", cv=10),
    # # # "cv_minmax": dict(method="minmax", cv=10),
    # "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50)),
    # # "jackknife_minmax_ab": dict(
    # #     method="minmax", cv=Subsample(n_resamplings=50)
    # # ),
   
    "cqr": dict(
        method="quantile", cv="split", alpha=alpha
    )
    }

#Slower strategies for debugging
# STRATEGIES = {
#       "jackknife_plus_ab": dict(method="plus", cv=Subsample(n_resamplings=50), conformity_score=GammaConformityScore()),# CHECK THIS WORKS! Gamma conformity score
# #     "conformal_quantile_regression": dict(
# #         method="quantile", cv="split", alpha=alpha
# #     )

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
training_data_name = ["sinex_het", "cube","sinex_con"] #Function data was generated with
if simulate_data:
    for name in training_data_name:
        plt.figure()
        plt.scatter(x_train[name], y_train[name], marker="x")
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
print("Beginning iteration")
#Run through each conformal strategy
if simulate_data:
    for strategy, params in STRATEGIES.items():
        #Run through each data generation technique
        for func_name in training_data_name:
            print(f"Performing {strategy} with {func_name} function")
            coverage,mean_width, cwc, y_ci, total_time = train_and_run_ci(mlp_regressor,x_train[func_name],y_train[func_name],x_test[func_name],y_test[func_name], alpha, func_name,strategy, params, real_data)
            results.append({
                "strategy": strategy,
                "data_name": func_name,
                "coverage": coverage,
                "mean_width": mean_width,
                "cwc": cwc,
                "total_time": total_time
            })
            y_cis[strategy][func_name] = y_ci
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
df.to_csv('results.csv', index=False)
#%%

if simulate_data:
    #Plot widths along x
    plot_widths_along_x(training_data_name, STRATEGIES, x_test,y_cis)
   
    #Plot coverage along x as hist
    num_bins = 11

    for func_name in training_data_name:
        heteroscedastic_coverage = get_coverage(
            y_test, y_cis, STRATEGIES, num_bins, x_test[func_name],func_name
        )
#%%
#Manually set bins if needed
# bins = [0, 1, 2, 3, 4, 5] # make so you don;t have to input the bins
# heteroscedastic_coverage = get_heteroscedastic_coverage(
#     y_test, y_cis, STRATEGIES, bins, x_test, "sinex_het"
# )

# fig = plt.figure()

data = pd.read_csv("results.csv")

# Set the width of the bars
bar_width = 0.4

# Iterate over each unique 'data_name'
for data_name in data['data_name'].unique():
    # Filter the data for the current 'data_name'
    df = data[data['data_name'] == data_name]
    
    # Create positions for the bars
    r1 = np.arange(len(df))
    r2 = [x + bar_width for x in r1]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot the bars for coverage on the primary y-axis
    ax1.bar(r1, df['coverage'], color='g', width=bar_width, edgecolor='grey', label='Coverage')
    
    # Plot the bars for mean width on the secondary y-axis
    ax2.bar(r2, df['mean_width'], color='b', width=bar_width, edgecolor='grey', label='Mean Width')
    
    # Add labels
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_ylabel('Coverage', fontweight='bold')
    ax2.set_ylabel('Mean Width',fontweight='bold')
    
    ax1.set_xticks([r + bar_width/2 for r in range(len(df))])
    ax1.set_xticklabels(df['strategy'])
    
    ax1.set_ylim(0, df['coverage'].max() * 1.1)
    ax2.set_ylim(0, df['mean_width'].max() * 1.1)
    
    # Add title and legend
    plt.title(f'Coverage and Mean Width for {data_name} function')
    ax1.axhline(1-alpha, ls="--", color="k")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Show the plot
    plt.show()