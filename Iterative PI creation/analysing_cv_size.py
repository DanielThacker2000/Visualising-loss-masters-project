# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:52:28 2024

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
file_name= "results_withcv.csv"
df = pd.read_csv(file_name)
df = df[df["cv_size"]!=1000]

more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="linear", noise_type2="normal", 
                               layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
                               x_lab="Number of CV models", y_var1="coverage",y_var2="mean_width",y_lab1="Coverage",y_lab2="Mean Width")

#%%
more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)")


















# df1 = pd.read_csv("combined_network_data.csv")
    
# df1 = df1[df1["strategy"]=="jackknife_plus"]
# df1 = df1[df1["num_samples"]==1000]
# df1.to_csv("justjkp_new.csv", index=True)