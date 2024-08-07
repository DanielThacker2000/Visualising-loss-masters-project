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

rns_df = df[df['strategy'] == 'split_resid_norm']

# Iterate through the rows where the strategy is 'CQR'
for index, row in df[df['strategy'] == 'CQR'].iterrows():
    # Find the matching row in the rns_df
    matching_row = rns_df[rns_df.drop(columns=['strategy', 'total_time']).eq(row.drop(['strategy', 'total_time'])).all(axis=1)]
    
    # If a matching row is found, update the 'total_time'
    if not matching_row.empty:
        df.at[index, 'total_time'] = matching_row['total_time'].values[0]


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