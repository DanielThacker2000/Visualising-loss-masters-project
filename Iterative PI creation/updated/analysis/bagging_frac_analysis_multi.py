# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:23:44 2024

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

file_name= "bagging_fracs_multi.csv"
df = pd.read_csv(file_name)
# df.loc[df["mean_width"] > 10000, "mean_width"] = None
# df.loc[df["normalised_mean_width"] > 1, "normalised_mean_width"] = None
df.loc[df["cwc"] < 0, "cwc"] = None
# df.loc[df["mwi"] > 10000, "mwi"] = None


hue = "num_features"
x_var = "bagging_fraction"
x_lab = "Bagging fraction"
y_var1 = "cwc"
y_var2 = "total_time"
y_lab1= "CWC"
y_lab2= "Total time (s)"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")
y_var1 = "coverage"
y_var2 = "normalised_mean_width"
y_lab1= "Coverage"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")
#%%

df_num10 = more_plotting.remove_var_df(df, "num_features", 10)

hue = "cv_size"
x_var = "bagging_fraction"
x_lab = "Bagging fraction"
y_var1 = "coverage"
y_var2 = "mwi"
y_lab1= "Coverage"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df_num10,df2=df_num10, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")

#%%

df_num10 = more_plotting.remove_var_df(df, "num_features", 10)

hue = "data_name"
x_var = "bagging_fraction"
x_lab = "Bagging fraction"
y_var1 = "coverage"
y_var2 = "normalised_mean_width"
y_lab1= "Coverage"
y_lab2= "Normalised Mean width"
more_plotting.plot_box_multi(df1=df_num10,df2=df_num10, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")