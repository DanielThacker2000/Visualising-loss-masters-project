# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:11:07 2024

@author: dan
"""
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.offsetbox import AnnotationBbox, TextArea
# import numpy as np
# import os
# import seaborn as sns
# import ast
# from PIL import Image
import more_plotting_functions as more_plotting

alpha= 0.1

file_name= "optimiser_more_functinos.csv"
df = pd.read_csv(file_name)

"Do stuff"

"""

ADD ADAM TO THIS GRAPH - AND LINEAR



TAKE REPEATS WTIH DIFFERENT NUMBER OF SAMPLES!!!!


"""

df_lbfgs = more_plotting.remove_var_df(df, "optimiser", "lbfgs")
df_sgd = more_plotting.remove_var_df(df, "optimiser", "sgd")
hue = "optimiser"
x_var = "data_name"
x_lab = "Function"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="total_time",y_lab1="CWC", y_lab2="Total time (s)",
                                title=None)

hue = "optimiser"
x_var = "data_name"
x_lab = "Function"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="mean_width",y_lab1="coverage", y_lab2="Mean width",
                                title=None)


#%%
hue = "optimiser"
x_var = "num_samples"
x_lab = "Number of samples"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="mean_width",y_lab1="coverage", y_lab2="Mean width",
                                title=None)


#%%
plotter = more_plotting.Plotter('base_path')
test_titles = ["Cube","Normal Mixture Model","SineX Constant", "SineX Heteroscedastic"]
plotter.plot_images_from_dir('functions for essay',test_titles,im_width=18,im_height=6,cols=2, titles=1)



# more_plotting.plot_coverage_for_specific_multi(df, strategy1="split", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
#                                                num_samples1=300, noise_std1=1,cv_size1= 1,x_var="num_samples",
#                                strategy2="split", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000)", 
#                                num_samples2=300, noise_std2=1,cv_size2= 1, x_lab="Number of training samples", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)")





