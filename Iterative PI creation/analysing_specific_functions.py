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

file_name= "all_data_simulated.csv"
df = pd.read_csv(file_name)



"LINEAR"

df_noise_norm = more_plotting.remove_var_df(df, "noise_type", "normal")
df_noise_stud = more_plotting.remove_var_df(df, "noise_type", "student")
df_noise_lap = more_plotting.remove_var_df(df, "noise_type", "laplace")

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



"SINEXHET AND CON"






"NMM"







"CUBE"