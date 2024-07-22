# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:12:42 2024

@author: dan
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os
import seaborn as sns
import ast
from PIL import Image
alpha=0.1

def apply_specific_order(filtered_df, x_var, specific_order):
    filtered_df = filtered_df.copy()
    filtered_df[x_var] = pd.Categorical(filtered_df[x_var], categories=specific_order, ordered=True)
    filtered_df.sort_values(by=x_var, inplace=True)
    return filtered_df

def plot_coverage_for_specific_multi_real(df, strategy1, data_name1, noise_type1, layer_size1, num_samples1, noise_std1, cv_size1, x_var,
                               strategy2, data_name2, noise_type2, layer_size2, num_samples2, noise_std2,cv_size2, x_lab, y_var1,y_var2,y_lab1, y_lab2,
                               title=None):
    
    def check_and_average(df, check_column):
        """
        Check for duplicate values in a specified column and average specified columns for duplicates.
    
        """
        # Check for duplicates
        duplicate_values = df[check_column].duplicated(keep=False)
        if not duplicate_values.any():
            return df  # No duplicates found
    
        # Group by the column with duplicate values and calculate means
        grouped = df[duplicate_values].groupby(check_column).mean()[[y_var1, y_var2]]
        
        # Create new rows with just the means
        new_rows = grouped.reset_index()
        
        # Drop the original duplicate rows
        df = df[~duplicate_values]
        # Append the new rows to the original DataFrame
        updated_df = pd.concat([df, new_rows], ignore_index=True)
        
        return updated_df

    # Function to filter dataframe based on specific values
    def filter_df(df, strategy, data_name, noise_type, layer_size, num_samples, noise_std,cv_size, x_var):
        filter_conditions = {
            'strategy': strategy,
            'dataset': data_name,
            'layer_size': layer_size,
            'cv_size': cv_size
        }

        # Remove the x_var from the filter conditions
        if x_var in filter_conditions:
            filter_conditions.pop(x_var)

        df.loc[df[y_var1] > 1000000, y_var1] = 0
        df.loc[df[y_var2] > 1000000, y_var2] = 0
        # Filter the dataframe based on the remaining conditions
        filtered_df = df
        keys = []
        for key, value in filter_conditions.items():
            keys.append(value)
            filtered_df = filtered_df[filtered_df[key] == value]
            # Group by the filtered conditions and calculate the mean for y_var1 and y_var2
        
        filtered_df = check_and_average(filtered_df, x_var)
        return filtered_df, keys
    
    # Filter the dataframes
    filtered_df1, keys1 = filter_df(df, strategy1, data_name1, noise_type1, layer_size1, num_samples1, noise_std1,cv_size1, x_var)
    filtered_df2, keys2 = filter_df(df, strategy2, data_name2, noise_type2, layer_size2, num_samples2, noise_std2,cv_size2, x_var)
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))
    
    bar_width = 0.4
    
    def get_y_lims(y_var):
        min1 = min(filtered_df1[y_var])
        min2 = min(filtered_df2[y_var])
        max1= max(filtered_df1[y_var])
        max2 = max(filtered_df2[y_var])
        mean_width_y_lim = min(min1,min2)
        mean_width_y_upper = max(max1,max2)
        return mean_width_y_lim,mean_width_y_upper
    
    y_var1_y_lim, y_var1_y_upper =  get_y_lims(y_var1)
    y_var2_y_lim, y_var2_y_upper =  get_y_lims(y_var2)
    
    # Function to plot coverage and mean width
    def plot_graph(ax, filtered_df, keys):
        # Positions of the bars on the x-axis
        bar_positions = range(len(filtered_df))
        
        
        
        if (x_var == "noise_std") and ("student" in keys):
            filtered_df = filtered_df.copy()
            filtered_df[x_var] = pd.Categorical(filtered_df[x_var], categories=sorted(filtered_df[x_var].unique(), reverse=True), ordered=True)
            filtered_df.sort_values(by=x_var, inplace=True)
            
            
        specific_order_layer = ['(200, 1000)', '(400, 2000)', '(200, 1000, 1000, 1000)', '(400, 2000, 2000, 2000)']
        specific_order_datasets = ["yacht_hydro", "boston","concrete","hour"]
        specific_order_strategy = ["split","split_gamma","split_resid_norm","jackknife_plus_ab"]

        if x_var == "layer_size":
            filtered_df = apply_specific_order(filtered_df, x_var, specific_order_layer)
            x_tick_labels = ['(200, 1000)', '(400, 2000)', '(200, 1000, 1000, 1000)', '(400, 2000, 2000, 2000)']
        elif x_var == "dataset":
            x_tick_labels = ["Yacht", "Boston","Concrete","Bike"]
            filtered_df = apply_specific_order(filtered_df, x_var, specific_order_datasets)
        elif x_var == "strategy":
            x_tick_labels = ["Split","Split Gamma","RNS","JK+ab"]
            filtered_df = apply_specific_order(filtered_df, x_var, specific_order_strategy)
        else:
            x_tick_labels = filtered_df[x_var]
        # Plot coverage as bars
        ax.bar(bar_positions, filtered_df[y_var1], width=bar_width, label=y_lab1, color='blue', align='center')
        ax.set_xlabel(x_lab, fontsize=16)
        ax.set_ylabel(y_lab1, fontsize=16)
        
        ax.set_xticks([p + bar_width / 2 for p in bar_positions])
        if x_var == "layer_size":
            ax.set_xticklabels(x_tick_labels, fontsize=16, rotation=15)
        else:
            ax.set_xticklabels(x_tick_labels, fontsize=16, rotation=0)
        ax.tick_params(axis='y', labelsize=16)
        if y_var1 == "coverage":
            ax.axhline(1-alpha, ls="--", color="k")
        
        # Create a second y-axis for mean_width
        ax2 = ax.twinx()
        ax2.bar([p + bar_width for p in bar_positions], filtered_df[y_var2], width=bar_width, label=y_lab2, color='red', align='center')
        ax2.set_ylabel(y_lab2, fontsize=16)
        ax2.tick_params(axis='y', labelsize=16)
        
        if y_var1 =="coverage" or y_var1 =="cwc":
            ax.set_ylim(y_var1_y_lim*0.9, 1)
            ax2.set_ylim(0,y_var2_y_upper*1.1)
        else:
            ax.set_ylim(y_var1_y_lim*0.9, y_var1_y_upper*1.1)
        
            ax2.set_ylim(y_var2_y_lim*0.9,y_var2_y_upper*1.1)
        # else:
        #     ax2.set_ylim(y_var2_y_lim,y_var2_y_upper*1.1)
        
        # Title and legend
        #ax.set_title(f'Coverage and Mean Width vs {x_var} for {keys[0]} function, {keys[1]} noise,\n {keys[2]} layer size, {keys[3]} sample number, {keys[4]} noise std')
        ax.legend(loc='upper left', fontsize=16)
        ax2.legend(loc='upper center', fontsize=16)

        ax.grid(True)
    
    # Plot the first graph
    plot_graph(ax1, filtered_df1, keys1)
    
    # Plot the second graph
    plot_graph(ax3, filtered_df2, keys2)
    #fig.suptitle(f'Coverage and Mean Width vs {x_var} for {keys1[0]} function, {keys1[1]} noise,\n {keys1[2]} layer size, {keys1[3]} sample number, {keys1[4]} noise std')
    #MANUALLY CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if not title:
        fig.suptitle(f'Comparing heteroscedastic vs constant noise for {keys1[1]}:{keys2[1]}, {keys1[0]}:{keys2[0]} strategy.',fontsize=16)
    else:
        fig.suptitle(title,fontsize=16)
    fig.tight_layout()
    plt.show()