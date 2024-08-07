# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:04:42 2024 FOR MULTI

@author: dan
"""
import pandas as pd
# from matplotlib.offsetbox import AnnotationBbox, TextArea
# import numpy as np
# import os
# import seaborn as sns
# import ast
# from PIL import Image
import more_plotting_functions as more_plotting

alpha= 0.1

file_name= "activ_func.csv"
df = pd.read_csv(file_name)
# df.loc[df["mean_width"] > 10000, "mean_width"] = None
df.loc[df["normalised_mean_width"] > 1, "normalised_mean_width"] = None
df.loc[df["cwc"] < 0, "cwc"] = None
df.loc[df["mwi"] > 10000, "mwi"] = None

df = more_plotting.remove_var_df(df,"strategy","jackknife_plus_ab")


hue = "data_name"
x_var = "active_function"
x_lab = "Activation Function"
y_var1 = "coverage"
y_var2 = "total_time"
y_lab1= "Coverage"
y_lab2= "Total time (s)"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Comparing activation function against fitting function for JK+ab")
y_var1 = "cwc"
y_var2 = "normalised_mean_width"
y_lab1= "CWC"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Comparing activation function against fitting function")

#%%
df_3 = more_plotting.remove_var_df(df,"num_features",3)
df_10 = more_plotting.remove_var_df(df,"num_features",10)

hue = "data_name"
x_var = "active_function"
x_lab = "Activation Function"
y_var1 = "coverage"
y_var2 = "coverage"
y_lab1= "Coverage"
y_lab2= "Coverage"
more_plotting.plot_box_multi(df1=df_3,df2=df_10, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")
y_var1 = "normalised_mean_width"
y_var2 = "normalised_mean_width"
y_lab1= "Normalised Mean width"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df_3,df2=df_10, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")

#%%
hue = "active_function"
x_var = "data_name"
x_lab = "Function"
y_var1 = "coverage"
y_var2 = "total_time"
y_lab1= "Coverage"
y_lab2= "Total time (s)"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Comparing activation function against fitting function for JK+ab")
y_var1 = "cwc"
y_var2 = "normalised_mean_width"
y_lab1= "CWC"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Comparing activation function against fitting function")