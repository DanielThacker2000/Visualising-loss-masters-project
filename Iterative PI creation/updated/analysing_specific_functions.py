# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:15:49 2024

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
plotter = more_plotting.Plotter('base_path')
file_name= "all_data_simulated_block.csv"
df = pd.read_csv(file_name)

df = df[df["strategy"] != "cv"]
df = df[df["strategy"] != "cv_plus"]
df.loc[df["cwc"] < 0, "cwc"] = None
# df = df[df["num_samples"] == 500]
# df = df[df["num_samples"] == 500]

"LINEAR"

df_noise_norm = more_plotting.remove_var_df(df, "noise_type", "normal")
df_noise_stud = more_plotting.remove_var_df(df, "noise_type", "student")
df_noise_lap = more_plotting.remove_var_df(df, "noise_type", "laplace")
df_noise_block = more_plotting.remove_var_df(df, "noise_type", "block")

hue = None
x_var = "noise_std"
x_lab = "Noise scaling factor"
more_plotting.plot_box_for_specific_multi(df1=df_noise_norm,df2=df_noise_stud, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="coverage",y_lab1="Coverage", y_lab2="Coverage",
                               title=None)

more_plotting.plot_box_for_specific_multi(df1=df_noise_norm,df2=df_noise_stud, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="mwi",y_var2="mwi",y_lab1="MWI", y_lab2="MWI",
                               title=None)


more_plotting.plot_box_for_specific_multi(df1=df_noise_norm,df2=df_noise_lap, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="coverage",y_lab1="Coverage", y_lab2="Coverage",
                               title=None)

#%%

hue = "noise_type"
x_var = "noise_std"
x_lab = "Strategy"
y_var1 = "coverage"
y_var2 = "cwc"
y_lab1= "Coverage"
y_lab2= "CWC"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)



#%%


hue = None
x_var = "noise_std"
x_lab = "Noise scaling factor"
more_plotting.plot_box_for_specific_multi(df1=df_noise_norm,df2=df_noise_stud, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="coverage",y_lab1="Coverage", y_lab2="Coverage",
                               title=None)

more_plotting.plot_box_for_specific_multi(df1=df_noise_lap,df2=df_noise_block, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="coverage",y_lab1="Coverage", y_lab2="Coverage",
                               title=None)


"SINEXHET AND CON"






"NMM"
#%%

hue = "layer_size"
x_var = "strategy"
x_lab = "Strategy"
y_var1 = "coverage"
y_var2 = "cwc"
y_lab1= "Coverage"
y_lab2= "CWC"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Coverage and CWC for different layer sizes for the NMM function")

hue = "layer_size"
x_var = "strategy"
x_lab = "Strategy"
y_var1 = "mean_width"
y_var2 = "mwi"
y_lab1= "Mean width"
y_lab2= "MWI"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)

"CUBE"

#%%

test_titles = ["Split","CQR","Gamma","RNS"]#Comparing optimiser against different multivariate functions
plotter.plot_images_from_dir('cube_function_graph',test_titles,im_width=16,im_height=6,cols=2, titles=1)
