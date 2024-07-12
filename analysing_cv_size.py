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
file_name= "all_data_simulated.csv"
df = pd.read_csv(file_name)
# df = df[df["cv_size"]!=1000]

more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="linear", noise_type2="normal", 
                               layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 50, 
                               x_lab="Number of CV models", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)",
                               title="Comparing strategies CV and CV+ coverage and computation time")

#%%
more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="cube", noise_type2="normal", layer_size2="(200, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="cwc",y_var2="mean_width",y_lab1="CWC",y_lab2="Mean width",
                               title="Comparing strategies CV and CV+ CWC and mean width")

#%%
more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv_plus", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000, 1000, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="cwc",y_var2="mean_width",y_lab1="CWC",y_lab2="Mean width",
                               title="Comparing CV+ (200, 1000) and (200, 1000, 1000, 1000) layer size CWC and mean width")



more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv_plus", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="cv_size",
                               strategy2="cv_plus", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000, 1000, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 50, x_lab="Number of CV models", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)",
                               title="Comparing CV+ (200, 1000) and (200, 1000, 1000, 1000) layer size coverage and computation time")

#%%


more_plotting.plot_coverage_for_specific_multi(df, strategy1="cv_plus", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=1000, noise_std1=1,cv_size1= 50,x_var="layer_size",
                               strategy2="jackknife_plus_ab", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000, 1000, 1000)", 
                               num_samples2=1000, noise_std2=1,cv_size2= 1, x_lab="Number of CV models", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)",
                               title="Comparing CV+ and JK+ab coverage and computation time")


more_plotting.plot_box_for_specific_function(df,x_var="layer_size", y_var="coverage", hue=None, y2_var="total_time", filter_function="cube", x_axis_name="Layer size", y_var_title1="Coverage",y_var_title2="Total time (s)",x2_var="layer_size")



test_titles = ["1","2","3","4"]
plotter = more_plotting.Plotter("results")
plotter.plot_images_from_dir('cv_analyse_plus_comp',test_titles,im_width=18,im_height=6,cols=1)


#%%
# df = pd.read_csv("all_data_simulated.csv")


# # Update the cv_size based on the strategy condition
# df.loc[df['strategy'].isin(['jackknife_plus', 'jackknife_plus_ab']), 'cv_size'] = df['num_samples']

# # Save the updated DataFrame to a new CSV file
# df.to_csv('teser_forall.csv', index=False)

# df.head()





# df1 = pd.read_csv("combined_network_data.csv")
    
# df1 = df1[df1["strategy"]=="jackknife_plus"]
# df1 = df1[df1["num_samples"]==1000]
# df1.to_csv("justjkp_new.csv", index=True)