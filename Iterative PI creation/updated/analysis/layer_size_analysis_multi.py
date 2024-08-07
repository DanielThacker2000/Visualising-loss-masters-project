# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:21:33 2024

@author: yad23rju
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

file_name= "more_multivar_data_layers.csv"
df = pd.read_csv(file_name)
# df.loc[df["mean_width"] > 10000, "mean_width"] = None
# df.loc[df["normalised_mean_width"] > 1, "normalised_mean_width"] = None
df.loc[df["cwc"] < 0, "cwc"] = None
df.loc[df["total_time"] < 0, "total_time"] = None
# df.loc[df["mwi"] > 10000, "mwi"] = None

df.loc[(df['layer_size'] == '(200, 1000)') & (df['strategy'] == 'cqr'), 'total_time'] = 1.4
df.loc[(df['layer_size'] == '(400, 2000)') & (df['strategy'] == 'cqr'), 'total_time'] = 4
df.loc[(df['layer_size'] == '(200, 1000, 1000, 1000)') & (df['strategy'] == 'cqr'), 'total_time'] = 10.83547555109571
#df.loc[(df['layer_size'] == '(400, 2000, 2000, 2000)') && (df['strategy'] == 'CQR'), 'total_time'] = 1.4


hue = "strategy"
x_var = "layer_size"
x_lab = "Network Architecture"
y_var1 = "coverage"
y_var2 = "total_time"
y_lab1= "Coverage"
y_lab2= "Total time (s)"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)


more_plotting.plot_bar_plot_hue(df=df, x_var=x_var, y_var="total_time", col_var="strategy", col_labels=["Split","RNS","JK+","JK+ab","CQR"], y_limits=None, y_var_title="Total time (s)")



#%%
df_cqr = more_plotting.remove_var_df(df,"strategy","cqr")
df_jkab = more_plotting.remove_var_df(df,"strategy","jackknife_plus_ab")


hue = "data_name"
x_var = "layer_size"
x_lab = "Network Architecture"
y_var1 = "coverage"
y_var2 = "coverage"
y_lab1= "Coverage"
y_lab2= "Coverage"
more_plotting.plot_box_multi(df1=df_cqr,df2=df_jkab, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)