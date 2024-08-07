# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:45:03 2024

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

file_name= "optimiser_for_multivariate.csv"
df = pd.read_csv(file_name)
# df.loc[df["mean_width"] > 10000, "mean_width"] = None
# df.loc[df["normalised_mean_width"] > 1, "normalised_mean_width"] = None
df.loc[df["cwc"] < 0, "cwc"] = None
# df.loc[df["mwi"] > 10000, "mwi"] = None

df = df[df['strategy'] == "jackknife_plus_ab"]

hue = "data_name"
x_var = "optimiser"
x_lab = "Optimiser"
y_var1 = "coverage"
y_var2 = "normalised_mean_width"
y_lab1= "Coverage"
y_lab2= "Normalised Mean Width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")

#%%
#more_plotting.plot_cat_plot(df=df,x_var="optimiser",y_var="coverage",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=(0.89,0.925))
more_plotting.plot_bar_plot_hue(df=df,x_var="optimiser",y_var="coverage",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=(0.89,0.925), y_var_title="Coverage")
more_plotting.plot_bar_plot_hue(df=df,x_var="optimiser",y_var="normalised_mean_width",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=None, y_var_title="Normalised mean width")

more_plotting.plot_bar_plot_hue(df=df,x_var="optimiser",y_var="total_time",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=None, y_var_title="Total time (s)")
#%%

df_3 = more_plotting.remove_var_df(df,"num_features",3)
df_10 = more_plotting.remove_var_df(df,"num_features",10)

hue = "data_name"
x_var = "optimiser"
x_lab = "Optimiser"
y_var1 = "total_time"
y_var2 = "total_time"
y_lab1= "Total time (s)"
y_lab2= "Total time (s)"
more_plotting.plot_box_multi(df1=df_3,df2=df_10, data_name1=None, hue1=hue, x_var=x_var,
                                data_name2=None, hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")






plotter = more_plotting.Plotter('base_path')
test_titles = ["Total time for different dataset dimensions",""]#Comparing optimiser against different multivariate functions
plotter.plot_images_from_dir('combine_plot',test_titles,im_width=16,im_height=6,cols=2, titles=1)







more_plotting.plot_bar_plot_hue(df=df_3,x_var="optimiser",y_var="total_time",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=None, y_var_title="Total time (s)")

more_plotting.plot_bar_plot_hue(df=df_10,x_var="optimiser",y_var="total_time",col_var="data_name", col_labels=["Linear","Polynomial","Non linear"], y_limits=None, y_var_title="Total time (s)")







# df1 = pd.read_csv("activ_func.csv")
    
# df1 = df1[df1["strategy"]=="jackknife_plus_ab"]
# df1 = df1[df1["num_samples"]==500]
# df1 = df1[df1["optimiser"]=="adam"]
# df1 = df1[df1["active_function"]=="relu"]
# df1.to_csv("justjkp_foradam.csv", index=True)