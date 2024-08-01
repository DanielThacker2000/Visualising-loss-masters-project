# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:55:59 2024

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


"ANALYSING GENERAL INFORMATION ABOUT THE STRATS - ASUMING NO CV_SIZE ITERATION AND NO BAGGING_FRAC OR OPTIMISER COLUMNS (DROP THEM)"

alpha= 0.1
#file_name= "combined_network_data_cv_adjusted.csv"#This is withou all the cv and cv_plus strategy data
file_name= "all_data_simulated.csv"
df = pd.read_csv(file_name)

THING_TO_ITERATE_OVER = "strategy"
   
   
#strategy, data_name, noise_type, layer_size, num_samples, noise_std ="jackknife_plus_ab","linear","normal","(200, 1000)",1000,1
x_var = THING_TO_ITERATE_OVER
# plot_coverage_for_specific(df, strategy="jackknife_plus_ab", data_name="cube", noise_type="normal", layer_size="(200, 1000)", num_samples=1000, noise_std=1, x_var="strategy")
# plot_coverage_for_specific(df, strategy="jackknife_plus_ab", data_name="linear", noise_type="normal", layer_size="(200, 1000)", num_samples=1000, noise_std=1, x_var="strategy")


# plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1, x_var="strategy",
#                                strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="student", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=4, x_lab="Strategy")


# plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1, x_var="noise_std",
#                                strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="student", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=4, x_lab="Noise Scaling Factor")

# plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=100, noise_std1=1, x_var="layer_size",
#                                strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1, x_lab="Layer Size")

# more_plotting.plot_coverage_for_specific_multi(df, strategy1="split_resid_norm", data_name1="sinex_het", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1,cv_size1= 1, x_var="strategy",
#                                 strategy2="split_resid_norm", data_name2="sinex_con", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1,cv_size2= 1, x_lab="strategy", y_var1="coverage",y_var2="mean_width")

more_plotting.plot_coverage_for_specific_multi(df, strategy1="split", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=300, noise_std1=1,cv_size1= 1,x_var="num_samples",
                               strategy2="split", data_name2="linear", noise_type2="normal", layer_size2="(200, 1000)", 
                               num_samples2=300, noise_std2=1,cv_size2= 1, x_lab="Number of training samples", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)")

more_plotting.plot_coverage_for_specific_multi(df, strategy1="jackknife_plus", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=300, noise_std1=1,cv_size1= 1,x_var="strategy",
                               strategy2="jackknife_plus", data_name2="linear", noise_type2="normal", layer_size2="(400, 2000)", 
                               num_samples2=300, noise_std2=1,cv_size2= 1, x_lab="Number of training samples", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)")

more_plotting.plot_coverage_for_specific_multi(df, strategy1="jackknife_plus", data_name1="linear", noise_type1="normal", layer_size1="(200, 1000)", 
                                               num_samples1=300, noise_std1=1,cv_size1= 1,x_var="strategy",
                               strategy2="jackknife_plus", data_name2="linear", noise_type2="normal", layer_size2="(400, 2000)", 
                               num_samples2=300, noise_std2=1,cv_size2= 1, x_lab="Number of training samples", y_var1="coverage",y_var2="total_time",y_lab1="Coverage",y_lab2="Total time (s)")
# plot_coverage_for_specific_multi(df, strategy1="split_resid_norm", data_name1="nmm", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=0.2, x_var="data_name",
#                                strategy2="split_resid_norm", data_name2="nmm", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1, x_lab="data_name")
# plot_coverage_for_specific_multi(df, strategy1="split_resid_norm", data_name1="sinex_het", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=0.2, x_var="layer_size",
#                                strategy2="split_resid_norm", data_name2="sinex_het", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1, x_lab="layer_size")
#%%

base_path = 'results'
function_name = "linear"
noise_type = "normal"
strategy = "jackknife_plus_ab"
num_samples_str = "1000"
std = "0.5"

iterated_parameter = THING_TO_ITERATE_OVER
parameter_list = ["(200, 1000)","(400, 2000)","(200, 1000, 1000, 1000)","(400, 2000, 2000, 2000)"]
parameter_list_strategy = ["jackknife_plus","split_resid_norm","cqr","jackknife_plus_ab"]
#parameter_list_strategy = ["split_resid_norm","cqr"]
if iterated_parameter == 'noise_std':
    match noise_type:
        case "normal":
            parameter_list = [0,0.2,0.5,1] #normal
        case "block":
            parameter_list = [0.01,0.2,0.5,1] #block normal
        case "student":
            parameter_list = [4,5,10,1000] #student t
        case "cauchy":
            parameter_list = [0.01,0.05,0.15,0.3] #Cauchy std
        case "laplace":
            parameter_list = [0.5,1,1.5,2] #Laplace
        case _:
            parameter_list = [100000] #failure
  
plotter = more_plotting.Plotter(base_path)
#plotter.plot_images(function_name, num_samples_str, noise_type, std, strategy, iterated_parameter,parameter_list)

# plotter.plot_images(function_name="linear", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
# plotter.plot_images(function_name="cube", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
# plotter.plot_images(function_name="sinex_het", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
plotter.plot_images(function_name="sinex_het", num_samples_str="1000", noise_type="normal", std="0.2", strategy="split_resid_norm",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
plotter.plot_images(function_name="sinex_het", num_samples_str="1000", noise_type="normal", std="0.5", strategy="split_resid_norm",layer_size="(200, 1000)", iterated_parameter="layer_size",parameter_list=parameter_list)

#%%
#plot_box(THING_TO_ITERATE_OVER,"coverage",None,"mean_width")

#plot_box_for_specific_function(df,THING_TO_ITERATE_OVER,"coverage",None,"mean_width", "linear",THING_TO_ITERATE_OVER, "Coverage","Mean Width",THING_TO_ITERATE_OVER)

#For linear and cube, plot the cov and width for each: strategy, num_samples, noise_std, noise_type
#Then on RD do the same for layer size boxplot somehow.


#%%

#LINEAR PLOTTING
x_var_array = ["strategy","num_samples", "layer_size"]
x_var_label_array = ["Strategy","Training data size","Netork layer size"]

#GOOD STUFF
# get_summary(x_var_array,x_var_label_array,df, filter_function="cube",y_var="coverage", y2_var="mean_width")
# get_summary(x_var_array,x_var_label_array, df,filter_function="cube",y_var="coverage", y2_var="mean_width")
more_plotting.get_summary(x_var_array,x_var_label_array,df, filter_function="linear",y_var="coverage", y2_var="total_time", y_var_title1="Coverage",y_var_title2="Time (s)")
# get_summary(x_var_array,x_var_label_array,df, filter_function="cube",y_var="mwi", y2_var="cwc", y_var_title1="Mean Winkler Score",y_var_title2="Coverage-Width Based Criterion")
# get_summary(x_var_array,x_var_label_array,df, filter_function="sinex_het",y_var="coverage", y2_var="mean_width", y_var_title1="Coverage",y_var_title2="Mean Width")
# get_summary(x_var_array,x_var_label_array,df, filter_function="sinex_het",y_var="mwi", y2_var="cwc", y_var_title1="Mean Winkler Score",y_var_title2="Coverage-Width Based Criterion")

# get_summary(x_var_array,x_var_label_array, df,filter_function="cube",y_var="coverage", y2_var="mean_width")

#%%

"""

ANALYSING NOISE TYPE

"""
hue = "strategy"
x_var = "noise_type"
x_lab = "Noise type"
y_var1 = "coverage"
y_var2 = "mean_width"
y_lab1= "Coverage"
y_lab2= "Mean width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)
y_var1 = "mwi"
y_var2 = "cwc"
y_lab1= "MWI"
y_lab2= "CWC"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="linear", hue1=hue, x_var=x_var,
                                data_name2="linear", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)
#%%

"""

ANALYSING LAYER SIZE

"""


df_cv_1 = more_plotting.remove_var_df(df, "cv_size", 1)
df_cv_100 = more_plotting.remove_var_df(df_cv_1, "num_samples", 100)
df_cv_1000 = more_plotting.remove_var_df(df_cv_1, "num_samples", 1000)


df_cv_1000.loc[df_cv_1000["total_time"] > 2000, "total_time"] = None

df_cv_100 = df_cv_100[df_cv_100["strategy"] != "split_resid_norm"]
df_cv_1000 = df_cv_1000[df_cv_1000["strategy"] != "split_resid_norm"]





df_cv_1000 = df_cv_1000[df_cv_1000["strategy"] != "cqr"]
df_cv_100 = df_cv_100[df_cv_100["strategy"] != "cqr"]


more_plotting.plot_box_for_specific_multi(df1=df_cv_1000,df2=df_cv_100, data_name1="linear", hue1="strategy", x_var="layer_size",
                                data_name2="linear", hue2="strategy",x_lab="Layer size", y_var1="coverage",y_var2="coverage",y_lab1="Coverage", y_lab2="Coverage",
                               title="1000 vs 100 samples for different layer sizes and strategies")

more_plotting.plot_box_for_specific_multi(df1=df_cv_1000,df2=df_cv_100, data_name1="linear", hue1="layer_size", x_var="strategy",
                                data_name2="linear", hue2="layer_size",x_lab="Layer size", y_var1="total_time",y_var2="total_time",y_lab1="Time (s)", y_lab2="Time (s)",
                               title=None)

#%%
"CUBE FUNCTION"

hue = "layer_size"
x_var = "strategy"
x_lab = "Strategy"
y_var1 = "mwi"
y_var2 = "mean_width"
y_lab1= "mwi"
y_lab2= "Mean width"
more_plotting.plot_box_multi(df1=df,df2=df, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)
#%%

"SINEX_HET and CON"
"TAKE MORE DATA"
# df_cv_1 = more_plotting.remove_var_df(df, "cv_size", 1)
# df_cv_100 = more_plotting.remove_var_df(df, "num_samples", 100)
df_het  = df[df["strategy"] != "cv_plus"]
df_het = df_het[df_het["strategy"] != "cv"]
hue = "num_samples"
x_var = "strategy"
x_lab = "Strategy"
y_var1 = "coverage"
y_var2 = "mean_width"
y_lab1= "coverage"
y_lab2= "Mean width"
more_plotting.plot_box_multi(df1=df_het,df2=df_het, data_name1="sinex_het", hue1=hue, x_var=x_var,
                                data_name2="sinex_het", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="SineX heteroscedastic and constant functions coverage and mean width")
more_plotting.plot_box_multi(df1=df_het,df2=df_het, data_name1="sinex_con", hue1=hue, x_var=x_var,
                                data_name2="sinex_con", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)
y_var2 = "coverage"
y_lab2= "Coverage"
more_plotting.plot_box_multi(df1=df_het,df2=df_het, data_name1="sinex_het", hue1=hue, x_var=x_var,
                                data_name2="sinex_con", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title="Comparing SineX heteroscedastic vs constant noise for different strategies and training data size")

#%%

"MAYBE GET RID OF NMM :("
df_nmm  = df[df["strategy"] != "cv_plus"]
df_nmm = df_het[df_het["strategy"] != "cv"]
hue = "layer_size"
x_var = "strategy"
x_lab = "Strategy"
y_var1 = "mwi"
y_var2 = "mean_width"
y_lab1= "mwi"
y_lab2= "Mean width"
more_plotting.plot_box_multi(df1=df_nmm,df2=df_nmm, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)

y_var1 = "coverage"
y_var2 = "cwc"
y_lab1= "Coverage"
y_lab2= "CWC"
more_plotting.plot_box_multi(df1=df_nmm,df2=df_nmm, data_name1="nmm", hue1=hue, x_var=x_var,
                                data_name2="nmm", hue2=hue,x_lab=x_lab, y_var1=y_var1,y_var2=y_var2,y_lab1=y_lab1, y_lab2=y_lab2,
                                title=None)
#%%
test_titles = ["1","2","3","4"]
plotter.plot_images_from_dir('test',test_titles,im_width=18,im_height=6,cols=1)

#%%
#PLOT SIZE STRATIFIED COVERAGE
more_plotting.plot_ssc(df, 'split_resid_norm', 'sinex_het', 0, 1000, 'normal', "(400, 2000, 2000, 2000)", "SSC Values for Residual Normal Score and heteroscedastic noise")
more_plotting.plot_ssc(df, 'cqr', 'sinex_het', 0, 1000, 'normal', '(200, 1000)', "SSC Values for CQR and heteroscedastic noise")
more_plotting.plot_ssc(df, 'jackkife_plus_ab', 'sinex_het', 0, 1000, 'normal', '(200, 1000)')
#plot_box_for_specific_function(df,x_var="", y_var=y_var, hue=None, y2_var=y2_var, filter_function=filter_function, x_axis_name=x_var_label, y_var_title1=y_var_title1,y_var_title2=y_var_title2,x2_var=x_var)
# noise_type = "normal"
#stds = [0,0.2,0.5,1]# [[1.         0.95       1.         1.         0.53333333]]

# coverage_for_each_std(file_name, noise_type, num_samples)
#more_plotting.coverage_for_each_strategy(file_name, stds, noise_type, 1000)
# [strategy,      data_name   ,coverage   ,mean_width ,cwc  ,mwi  ,total_time ,noise_std  ,num_samples      ,noise_type ,layer_size ,ssc]
#[strategy,      data_name  , noise_std  ,num_samples      ,noise_type ,layer_size ,ssc]