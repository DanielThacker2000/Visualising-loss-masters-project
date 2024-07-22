# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:57:16 2024

@author: dan
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
import numpy as np
import os
import seaborn as sns
import ast
from PIL import Image
import more_plotting_functions as more_plotting
import plotting_real_dataset as plotting_real

alpha= 0.1
file_name= "real_results_boston_yacht_hour_concrete.csv"
df = pd.read_csv(file_name)
df.drop(columns="datset")

"""


BOSTON VS BIKE DATASET COV, WIDTH , TIME AND MWI



"""
plotting_real.plot_coverage_for_specific_multi_real(
    df, 
    strategy1="jackknife_plus_ab", data_name1="boston", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1,cv_size1= 50,
    x_var="strategy",
    strategy2="jackknife_plus_ab", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
    x_lab="Strategy", y_var1="coverage",y_var2="mean_width",y_lab1="Coverage",y_lab2="Mean width",
    title="Boston data set (506 data points) vs Bike data set (17379 data points)"
    )

plotting_real.plot_coverage_for_specific_multi_real(
    df, 
    strategy1="jackknife_plus_ab", data_name1="boston", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1,cv_size1= 50,
    x_var="strategy",
    strategy2="jackknife_plus_ab", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
    x_lab="Strategy", y_var1="mwi",y_var2="total_time",y_lab1="MWI",y_lab2="Total time (s)",
    title="Boston data set (506 data points) vs Bike data set (17379 data points)"
    )
#%%

# plotting_real.plot_coverage_for_specific_multi_real(df, strategy1="jackknife_plus_ab", data_name1="yacht_hydro", noise_type1="normal", layer_size1="(200, 1000)", 
#                                                num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="dataset",
#                                strategy2="jackknife_plus_ab", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", 
#                                num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="cwc",y_var2="total_time",y_lab1="CWC",y_lab2="Time (s)",
#                                title="Comparing data set CWC and computation time")
"""


APPENDIX THING LOOKING AT HOW WEIRD CWC IS


"""

plotting_real.plot_coverage_for_specific_multi_real(df, strategy1="jackknife_plus_ab", data_name1="yacht_hydro", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="dataset",
                               strategy2="split", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="cwc",y_var2="total_time",y_lab1="CWC",y_lab2="Time (s)",
                               title="Comparing JK+ab and Split method for different data sets")
plotting_real.plot_coverage_for_specific_multi_real(df, strategy1="jackknife_plus_ab", data_name1="yacht_hydro", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="dataset",
                               strategy2="split", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="cwc",y_var2="mean_width",y_lab1="CWC",y_lab2="Mean Width",
                               title="Comparing JK+ab and Split method for different data sets")
#%%
"""

LAYER SIZE

"""
# plotting_real.plot_coverage_for_specific_multi_real(
#     df, 
#     strategy1="jackknife_plus_ab", data_name1="boston", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1,cv_size1= 50,
#     x_var="strategy",
#     strategy2="split_gamma", data_name2="boston", noise_type2="normal", layer_size2="(400, 2000, 2000, 2000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
#     x_lab="Number of CV models", y_var1="coverage",y_var2="mean_width",y_lab1="Coverage",y_lab2="Mean width",
#     title="Yacht data set (308 data points) vs Bike data set (17379 data points)"
#     )
#%%

"""

SUMMARTIVE OF ALL DATASETS

TAKE NORMALISED MPIW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


"""


plotting_real.plot_coverage_for_specific_multi_real(
    df, 
    strategy1="jackknife_plus_ab", data_name1="boston", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1,cv_size1= 50,
    x_var="dataset",
    strategy2="split_resid_norm", data_name2="hour", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
    x_lab="Strategy", y_var1="coverage",y_var2="mwi",y_lab1="Coverage",y_lab2="MWI",
    title="All datasets"
    )






#%%
test_titles = ["Boston data set (506 data points) vs Bike data set (17379 data points)",""]
# test_titles = ["sdjfknsd","Predicted against target value JK+ab","Predicted against target value Gamma Score","Coverage for each quantile JK+ab","Coverage for each quantile Gamma Score","Width against target value JK+ab","Width against target value Gamma Score"]
plotter = more_plotting.Plotter("results")
plotter.plot_images_from_dir('real_data_graphs',test_titles,im_width=18,im_height=6,cols=1,titles=1)