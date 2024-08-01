# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 08:47:51 2024

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

file_name= "multivar_data.csv"
df = pd.read_csv(file_name)
df.loc[df["mean_width"] > 10000, "mean_width"] = None
df.loc[df["normalised_mean_width"] > 1, "normalised_mean_width"] = None
df.loc[df["cwc"] < 0, "cwc"] = None
df.loc[df["mwi"] > 10000, "mwi"] = None



#%%
"Coverage and normalised mean width, with large outliers removed for split_resid. Comparing number of samples and strategy. Clearly JK+ab performs the best"
hue = "num_samples"
x_var = "strategy"
x_lab = "Strategy"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="normalised_mean_width",y_lab1="coverage", y_lab2="Mean width",
                                title=None)
#%%
"CWC and MWI, with large outliers removed for split_resid. Comparing number of samples and strategy. Clearly JK+ab performs the best. Negative CWC scores removed"
hue = "num_samples"
x_var = "strategy"
x_lab = "Strategy"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="mwi",y_lab1="CWC", y_lab2="MWI",
                                title=None)

#%%
"Analysing dimension size "
hue = "num_features"
x_var = "strategy"
x_lab = "Strategy"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="mwi",y_lab1="CWC", y_lab2="MWI",
                                title=None)
#more_plotting.plot_ssc(df, 'split_resid_norm', 'linear_multi', 0, 1000, 'normal', "(200, 1000)", "SSC Values for Residual Normal Score and heteroscedastic noise")

#%%
hue = "strategy"
x_var = "data_name"
x_lab = "Function"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="total_time",y_lab1="Coverage", y_lab2="Total time (s)",
                                title=None)

#%%

"NOW WITH MORE DATA - LAYER SIZE COMPARISON"
hue = "strategy"
x_var = "layer_size"
x_lab = "Layer Size"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="total_time",y_lab1="Coverage", y_lab2="Total time (s)",
                                title="Coverage and time for linear multivariate function")

"Outliers are reduced in NMW for further layer size"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="normalised_mean_width",y_lab1="CWC", y_lab2="Normalised Mean Width",
                                title="CWC and NWM for linear multivariate function")



df_3 = more_plotting.remove_var_df(df,"num_features",3)
df_10 = more_plotting.remove_var_df(df,"num_features",10)

#%%

"Outliers are reduced in NMW for further layer size"
more_plotting.plot_box_multi(df1=df_3,df2=df_10, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="cwc",y_var2="cwc",y_lab1="CWC", y_lab2="CWC",
                                title="Linear multivariate function with 3 variables vs 10 variables CWC")



#%%

"WOW NOISE DOES NOTHIGN"
hue = "strategy"
x_var = "noise_std"
x_lab = "Normal noise STD"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear_multi", hue1=hue, x_var=x_var,
                                data_name2="linear_multi", hue2=hue,x_lab=x_lab, y_var1="coverage",y_var2="total_time",y_lab1="Coverage", y_lab2="Total time (s)",
                                title="Coverage and time for linear multivariate function")
