# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 11:45:17 2024

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

alpha= 0.1
#file_name= "boostrap_cv_size.csv" THIS ONE HAS DIFFERENT FUNCTIONS THAN LINEAR IN
file_name= "bootstrap_cv_size_optimersers_fro_jkpab.csv"
df = pd.read_csv(file_name)

more_plotting.plot_coverage_for_specific_multi(
    df, 
    strategy1="jackknife_plus_ab", data_name1="linear", noise_type1="normal", layer_size1="(400, 2000, 2000, 2000)", num_samples1=1000, noise_std1=1,cv_size1= 50,
    x_var="optimiser",
    strategy2="jackknife_plus_ab", data_name2="linear", noise_type2="normal", layer_size2="(400, 2000, 2000, 2000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
    x_lab="Number of CV models", y_var1="coverage",y_var2="mean_width",y_lab1="Coverage",y_lab2="Mean width",
    title="Yacht data set (308 data points) vs Bike data set (17379 data points)"
    )
#%%

"""
Comparing cv size 5 and 50 (left and right) for different optimisers 


"""
df_cv_five = more_plotting.remove_var_df(df, "cv_size", 5)
df_cv_fifty = more_plotting.remove_var_df(df, "cv_size", 50)
more_plotting.plot_box_for_specific_function(df_cv_five,"optimiser","coverage","bagging_fraction","total_time", "linear","optimiser", "Coverage","Time (s)","optimiser")
more_plotting.plot_box_for_specific_function(df_cv_fifty,"optimiser","coverage","bagging_fraction","total_time", "linear","optimiser", "Coverage","Time (s)","optimiser")



#%%
"""
ONLY ADAM AND BAGGING FRAC 0.1, LOOKING AT TIME AND CV SIZE
"""
df_optmisier_adam = more_plotting.remove_var_df(df, "optimiser", "adam")
df_optmisier_adam = more_plotting.remove_var_df(df_optmisier_adam, "bagging_fraction", 0.1)


more_plotting.plot_box_for_specific_function(df_optmisier_adam,"cv_size","coverage",None,"total_time", "linear","CV Size", "Coverage","Time (s)","cv_size")

#%%

