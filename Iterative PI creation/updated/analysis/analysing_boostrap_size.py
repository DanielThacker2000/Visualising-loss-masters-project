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
#file_name= "bootstrap_cv_size.csv" #THIS ONE HAS DIFFERENT FUNCTIONS THAN LINEAR IN (only jkpab)
file_name= "bootstrap_cv_size_optimersers_fro_jkpab.csv" #Only linear justjkp_forbagging HAS MORE DATA - ADD TO IT

df = pd.read_csv(file_name)
df = df[df['layer_size'] == "(400, 2000, 2000, 2000)"]

df_bag_frac = more_plotting.remove_var_df(df, "bagging_fraction", 0.1)
#df_cv_fifty = more_plotting.remove_var_df(df, "cv_size", 50)

more_plotting.plot_coverage_for_specific_multi(
    df_bag_frac, 
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
#PLOTTING TIMES AND COVERAGES FOR CV SIZE 5 AND 50 MANUALLY NEXT TO EACH OTHER RATHER THAN DIFFERENT
# more_plotting.plot_box_single(df=df_cv_five, x_var="optimiser", y_var="coverage", hue="bagging_fraction", filter_function="linear",
#                               y_var_title="Coverage",x_var_title="Optimiser",legend_title="Bagging Fraction", title="CV size 5")
# more_plotting.plot_box_single(df=df_cv_fifty, x_var="optimiser", y_var="coverage", hue="bagging_fraction", filter_function="linear",
#                               y_var_title="Coverage",x_var_title="Optimiser",legend_title="Bagging Fraction", title="CV size 50")
# more_plotting.plot_box_single(df=df_cv_five, x_var="optimiser", y_var="total_time", hue="bagging_fraction", filter_function="linear",
#                               y_var_title="Total time (s)",x_var_title="Optimiser",legend_title="Bagging Fraction", title="CV size 5")
# more_plotting.plot_box_single(df=df_cv_fifty, x_var="optimiser", y_var="total_time", hue="bagging_fraction", filter_function="linear",
#                               y_var_title="Total time (s)",x_var_title="Optimiser",legend_title="Bagging Fraction", title="CV size 50")




#%%
hue = "optimiser"
x_var = "bagging_fraction"
x_lab = "Bagging fraction"
more_plotting.plot_box_for_specific_multi(df1=df_cv_five,df2=df_cv_fifty, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="total_time",y_var2="total_time",y_lab1="Time (s)", y_lab2="Time (s)",
                               title=None)

more_plotting.plot_box_for_specific_multi(df1=df_cv_five,df2=df_cv_fifty, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="coverage",
                                y_lab1="Coverage", y_lab2="Coverage",
                               title=None)

more_plotting.plot_box_for_specific_multi(df1=df_cv_five,df2=df_cv_fifty, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="cwc",y_lab1="CWC", y_lab2="CWC",
                               title="Comparing enemble size CWC for different optimisers and bagging fractions")
#%%
"""
ONLY ADAM AND BAGGING FRAC 0.1, LOOKING AT TIME AND CV SIZE
"""
df_optmisier_adam = more_plotting.remove_var_df(df, "optimiser", "adam")
df_optmisier_adam = more_plotting.remove_var_df(df_optmisier_adam, "bagging_fraction", 0.1)
more_plotting.plot_box_for_specific_function(df_optmisier_adam,"cv_size","coverage",None,"total_time", "linear","CV Size", "Coverage","Time (s)","cv_size")

#%%
plotter = more_plotting.Plotter('base_path')
test_titles = ["Comparing bagging fractions between different optimisers for ensemble size 5 and 50",""]
plotter.plot_images_from_dir('cv_analyais_for_rep/combine_plot_2',test_titles,im_width=18,im_height=6,cols=1, titles=1)