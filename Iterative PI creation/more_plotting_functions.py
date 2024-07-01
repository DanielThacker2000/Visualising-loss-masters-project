# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:36:54 2024

@author: dan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 08:58:48 2024

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
alpha= 0.1
file_name= "combined_network_data.csv"
df = pd.read_csv(file_name)
x_var = "noise_std"


class Plotter:
    def __init__(self, base_path):
        self.base_path = base_path
   
    def load_images(self, function_name, num_samples, noise_type, std, strategy, iterated_parameter, parameter_list):
        """
        Load images from the specified directories.
       
        Parameters:
        - function_name (str): The name of the function.
        - num_samples (str): The number of samples.
        - noise_type (str): The type of noise.
        - parameter_list (list): A list of values for the iterated parameter.
        - strategy (str): The strategy name.
        - iterated_parameter (str): The name of the parameter being iterated over.
       
        Returns:
        - images (list): A list of tuples containing the iterated parameter value and the corresponding image.
        """
        images = []
       
        for param in parameter_list:

            param_str = str(param)
            # Create directory path based on the iterated parameter
            if iterated_parameter == 'num_samples':
                dir_path = os.path.join(self.base_path, function_name, param_str, noise_type, std, strategy)
                param_list_for_title = [function_name, noise_type, std, strategy]
            elif iterated_parameter == 'noise_type':
                dir_path = os.path.join(self.base_path, function_name, num_samples, param_str, std, strategy)
                param_list_for_title = [function_name, num_samples, std, strategy]
            elif iterated_parameter == 'noise_std':
                dir_path = os.path.join(self.base_path, function_name, num_samples, noise_type, param_str, strategy)
                param_list_for_title = [function_name, num_samples, noise_type, strategy]
            elif iterated_parameter == 'strategy':
                dir_path = os.path.join(self.base_path, function_name, num_samples, noise_type, std, param_str)
                param_list_for_title = [function_name, num_samples, noise_type, std]
            elif iterated_parameter == 'function_name':
                dir_path = os.path.join(self.base_path, param_str, num_samples, noise_type, std, strategy)
                param_list_for_title = [num_samples, noise_type, std, strategy]
            else:
                dir_path = os.path.join(self.base_path, function_name, num_samples, noise_type, std, strategy)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".png"):
                        img_path = os.path.join(dir_path, file)
                        img = Image.open(img_path)
                        images.append((param, img))
            else:
                print(f"Directory does not exist: {dir_path}")
        return images,param_list_for_title
   
    def plot_images(self, function_name, num_samples_str, noise_type, std, strategy,layer_size, iterated_parameter,parameter_list):
        """
        Plot the loaded images in a formatted subplot.
       
        Parameters:
        - function_name (str): The name of the function.
        - num_samples (str): The number of samples.
        - noise_type (str): The type of noise.
        - parameter_list (list): A list of values for the iterated parameter.
        - strategy (str): The strategy name.
        - iterated_parameter (str): The name of the parameter being iterated over.
        """
        filter_conditions = {
            'strategy': strategy,
            'Function Name': function_name,
            'Noise Type': noise_type,
            'Layer Size': layer_size,
            'Number Samples': num_samples_str,
            'Noise scale factor': std
        }
       
        #remove the x_var from the filter conditions
        if iterated_parameter in filter_conditions:
            filter_conditions.pop(iterated_parameter)
           
        keys=[]
        values= []
        for key, value in filter_conditions.items():
            keys.append(key)
            values.append(value)
            #print(key, value)
           

        images, param_list_for_title = self.load_images(function_name, num_samples_str, noise_type, std, strategy, iterated_parameter,parameter_list)

        if not images:
            print("No images to plot.")
            return
       
        num_images = len(images)
        cols = 2  # Number of columns in the subplot
        rows = (num_images // cols) + (num_images % cols > 0)
        plt.clf()
        fig, axes = plt.subplots(rows, cols, figsize=(24, 12 * rows))
        plt.subplots_adjust(wspace=0, hspace=0.05, left=0, right=1, bottom=0, top=1)
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
       
        for ax in axes:
            ax.axis('off')  # Hide the axes by default
           
        for idx, (param, img) in enumerate(images):
            ax = axes[idx]
            ax.imshow(img, aspect='auto')
            ax.set_title(f"{iterated_parameter}: {param}", fontsize=40, pad=0)
            ax.axis('off')  # Show the axes
       
        plt.suptitle(f"{keys[0]} : {values[0]} - {keys[1]} : {values[1]} - {keys[2]} : {values[2]} - {keys[3]} : {values[3]}", fontsize=45, y=1.05)
        # fig.text(0.5, -0.02, 'x', ha='center', va='center', fontsize=50)
        # fig.text(-0.03, 0.5, 'y', ha='center', va='center', rotation='vertical', fontsize=50)
        plt.show()
       
       
    def plot_images_from_dir(self, dir_path,title_list):
        images = []
        if len(title_list) != len(os.listdir(dir_path)):
            print("WRONG SIZE TITLE LIST!")
        for title,file in zip(title_list,os.listdir(dir_path)):
            if os.path.exists(dir_path):
                if file.endswith(".png"):
                    img_path = os.path.join(dir_path, file)
                    img = Image.open(img_path)
                    images.append((title, img))
            else:
                print(f"Directory does not exist: {dir_path}")
               
        if not images:
            print("No images to plot.")
            return
       
        num_images = len(images)
        cols = 2  # Number of columns in the subplot
        rows = (num_images // cols) + (num_images % cols > 0)
        plt.clf()
        fig, axes = plt.subplots(rows, cols, figsize=(40, 12 * rows))
        plt.subplots_adjust(wspace=0, hspace=0.05, left=0, right=1, bottom=0, top=1)
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
       
        for ax in axes:
            ax.axis('off')  # Hide the axes by default
           
        for idx, (param, img) in enumerate(images):
            ax = axes[idx]
            ax.imshow(img, aspect='auto')
            ax.axis('off')  # Show the axes
       
        plt.show()
       
def coverage_for_each_strategy(file_name, stds, noise_type, num_samples):
    data = pd.read_csv(file_name)

    # Set the width of the bars
    bar_width = 0.4
    for std in stds:
        # Iterate over each unique 'data_name'
        for data_name in data['data_name'].unique():
            # Filter the data for the current 'data_name'
            df = data[(data['data_name'] == data_name) & (data['noise_std'] == std)]
            if df.empty == True:
                print("EMPTY DF!")
                break

            # Create positions for the bars
            r1 = np.arange(len(df))
            r2 = [x + bar_width for x in r1]
           
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
           
            # Plot the bars for coverage on the primary y-axis
            ax1.bar(r1, df['coverage'], color='g', width=bar_width, edgecolor='grey', label='Coverage')
           
            # Plot the bars for mean width on the secondary y-axis
            ax2.bar(r2, df['mean_width'], color='b', width=bar_width, edgecolor='grey', label='Mean Width')
           
            # Add labels
            ax1.set_xlabel('Strategy', fontweight='bold')
            ax1.set_ylabel('Coverage', fontweight='bold')
            ax2.set_ylabel('Mean Width',fontweight='bold')
           
            ax1.set_xticks([r + bar_width/2 for r in range(len(df))])
            ax1.set_xticklabels(df['strategy'], rotation=45)
           
            ax1.set_ylim(0.75, df['coverage'].max() * 1.1)
            ax2.set_ylim(0.75, df['mean_width'].max() * 1.1)
           
            # Add title and legend
            plt.title(f'Coverage and Mean Width for {data_name} function with noise standard deviation {std}')
            ax1.axhline(1-alpha, ls="--", color="k")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            base_path = "results_strategy_coverage"
            # save_path = create_save_directory("coverage_against_strategy.png", data_name,num_samples, noise_type, std, "pointless",base_path)
            # plt.savefig(save_path)
            # Show the plot
            plt.show()
           
def coverage_for_each_std(file_name, noise_type, num_samples):
    data = pd.read_csv(file_name)

    # Set the width of the bars
    bar_width = 0.4
    for stategy in data['strategy'].unique():
    # Iterate over each unique 'data_name'
        for data_name in data['data_name'].unique():
            # Filter the data for the current 'data_name'
            df = data[(data['data_name'] == data_name) & (data['strategy'] == stategy)]
            if df.empty == True:
                break
           
            # Create positions for the bars
            r1 = np.arange(len(df))
            r2 = [x + bar_width for x in r1]
           
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
           
            # Plot the bars for coverage on the primary y-axis
            ax1.bar(r1, df['coverage'], color='g', width=bar_width, edgecolor='grey', label='Coverage')
           
            # Plot the bars for mean width on the secondary y-axis
            ax2.bar(r2, df['mean_width'], color='b', width=bar_width, edgecolor='grey', label='Mean Width')
           
            # Add labels
            ax1.set_xlabel('Noise standard deviation', fontweight='bold')
            ax1.set_ylabel('Coverage', fontweight='bold')
            ax2.set_ylabel('Mean Width',fontweight='bold')
           
            ax1.set_xticks([r + bar_width/2 for r in range(len(df))])
            ax1.set_xticklabels(df['noise_std'], rotation=45)
           
            ax1.set_ylim(0, df['coverage'].max() * 1.1)
            ax2.set_ylim(0, df['mean_width'].max() * 1.1)
           
            # Add title and legend
            plt.title(f'Coverage and Mean Width for {data_name} function and {stategy} strategy')
            ax1.axhline(1-alpha, ls="--", color="k")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            base_path = "results_std_coverage"
           
            # save_path = create_save_directory("coverage_against_std.png", data_name,num_samples, noise_type, "pointless", stategy,base_path)
            # plt.savefig(save_path)
            # Show the plot
            plt.show()

# Function to plot coverage
def plot_coverage_for_specific(df, strategy, data_name, noise_type, layer_size, num_samples, noise_std, x_var):
    #filter dataframe based on the specific values
    filter_conditions = {
        'strategy': strategy,
        'data_name': data_name,
        'noise_type': noise_type,
        'layer_size': layer_size,
        'num_samples': num_samples,
        'noise_std': noise_std
    }
   
    #remove the x_var from the filter conditions
    if x_var in filter_conditions:
        filter_conditions.pop(x_var)
   
    #filter the dataframe based on the remaining conditions
    filtered_df = df
    keys=[]
    for key, value in filter_conditions.items():
        keys.append(value)
        filtered_df = filtered_df[filtered_df[key] == value]

    fig, ax1 = plt.subplots(figsize=(10, 6))
   
    bar_width = 0.4
   
    # Positions of the bars on the x-axis
    bar_positions = range(len(filtered_df))
   
    # Plot coverage as bars
    ax1.bar(bar_positions, filtered_df['coverage'], width=bar_width, label='Coverage', color='blue', align='center')
    ax1.set_xlabel(x_var)
    ax1.set_ylabel('Coverage')
    ax1.set_xticks([p + bar_width / 2 for p in bar_positions])
    ax1.set_xticklabels(filtered_df[x_var])
    ax1.tick_params(axis='y')
    ax1.axhline(1-alpha, ls="--", color="k")
    # Create a second y-axis for mean_width
    ax2 = ax1.twinx()
    ax2.bar([p + bar_width for p in bar_positions], filtered_df['mean_width'], width=bar_width, label='Mean Width', color='red', align='center')
    ax2.set_ylabel('Mean Width')
    ax2.tick_params(axis='y')
    
    ax1.set_ylim(0.6, df['coverage'].max() * 1.1)
    #ax2.set_ylim(0.6, df['mean_width'].max() * 1.1)
    # Title and legend
    plt.title(f'Coverage and Mean Width vs {x_var} for {keys[0]} function, {keys[1]} noise, {keys[2]} layer size, {keys[3]} sample number, {keys[4]} noise std')
    fig.tight_layout()

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # Display the plot
    plt.grid(True)
    plt.show()

def plot_coverage_for_specific_multi(df, strategy1, data_name1, noise_type1, layer_size1, num_samples1, noise_std1, x_var,
                               strategy2, data_name2, noise_type2, layer_size2, num_samples2, noise_std2, x_lab):
    # Function to filter dataframe based on specific values
    def filter_df(df, strategy, data_name, noise_type, layer_size, num_samples, noise_std, x_var):
        filter_conditions = {
            'strategy': strategy,
            'data_name': data_name,
            'noise_type': noise_type,
            'layer_size': layer_size,
            'num_samples': num_samples,
            'noise_std': noise_std
        }

        # Remove the x_var from the filter conditions
        if x_var in filter_conditions:
            filter_conditions.pop(x_var)

        # Filter the dataframe based on the remaining conditions
        filtered_df = df
        keys = []
        for key, value in filter_conditions.items():
            keys.append(value)
            filtered_df = filtered_df[filtered_df[key] == value]
        
        return filtered_df, keys
    
    # Filter the dataframes
    filtered_df1, keys1 = filter_df(df, strategy1, data_name1, noise_type1, layer_size1, num_samples1, noise_std1, x_var)
    filtered_df2, keys2 = filter_df(df, strategy2, data_name2, noise_type2, layer_size2, num_samples2, noise_std2, x_var)
    
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 6))
    
    bar_width = 0.4
    
    # Function to plot coverage and mean width
    def plot_graph(ax, filtered_df, keys):
        # Positions of the bars on the x-axis
        bar_positions = range(len(filtered_df))
        
        
        
        if (x_var == "noise_std") and ("student" in keys):
            filtered_df = filtered_df.copy()
            filtered_df[x_var] = pd.Categorical(filtered_df[x_var], categories=sorted(filtered_df[x_var].unique(), reverse=True), ordered=True)
            filtered_df.sort_values(by=x_var, inplace=True)
            
            
        specific_order_layer = ['(200, 1000)', '(400, 2000)', '(200, 1000, 1000, 1000)', '(400, 2000, 2000, 2000)']
        if x_var == "layer_size":
            filtered_df = filtered_df.copy()
            filtered_df[x_var] = pd.Categorical(filtered_df[x_var], categories=specific_order_layer, ordered=True)
            filtered_df.sort_values(by=x_var, inplace=True)
        # Plot coverage as bars
        ax.bar(bar_positions, filtered_df['coverage'], width=bar_width, label='Coverage', color='blue', align='center')
        ax.set_xlabel(x_lab, fontsize=16)
        ax.set_ylabel('Coverage', fontsize=16)
        
        ax.set_xticks([p + bar_width / 2 for p in bar_positions])
        if x_var == "layer_size":
            ax.set_xticklabels(filtered_df[x_var], fontsize=16, rotation=15)
        else:
            ax.set_xticklabels(filtered_df[x_var], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.axhline(1-alpha, ls="--", color="k")
        
        # Create a second y-axis for mean_width
        ax2 = ax.twinx()
        ax2.bar([p + bar_width for p in bar_positions], filtered_df['mean_width'], width=bar_width, label='Mean Width', color='red', align='center')
        ax2.set_ylabel('Mean Width', fontsize=16)
        ax2.tick_params(axis='y', labelsize=16)

        ax.set_ylim(0.6, 1)
        
        # Title and legend
        #ax.set_title(f'Coverage and Mean Width vs {x_var} for {keys[0]} function, {keys[1]} noise,\n {keys[2]} layer size, {keys[3]} sample number, {keys[4]} noise std')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True)
    
    # Plot the first graph
    plot_graph(ax1, filtered_df1, keys1)
    
    # Plot the second graph
    plot_graph(ax3, filtered_df2, keys2)
    #fig.suptitle(f'Coverage and Mean Width vs {x_var} for {keys1[0]} function, {keys1[1]} noise,\n {keys1[2]} layer size, {keys1[3]} sample number, {keys1[4]} noise std')
    #MANUALLY CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fig.suptitle(f'Comparing normal and student noise for {keys1[1]} function, {keys1[2]} vs {keys2[2]} noise type, {keys1[3]} sample number, {keys1[0]} strategy.',fontsize=16)
    fig.tight_layout()
    plt.show()
    
def plot_box(x_var, y_var, hue, y2_var):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
   
    # First boxplot
    sns.boxplot(data=df, x=x_var, y=y_var, hue=hue, ax=ax1)
    ax1.axhline(1-alpha, ls="--", color="k")
    ax1.set_title("Training data size against coverage")
    ax1.set_xlabel(x_var)
    ax1.set_ylabel("Coverage")
   
    # Second boxplot
    sns.boxplot(data=df, x=x_var, y=y2_var, hue=hue, ax=ax2)
    ax2.set_title("Training data size against mean width")
    ax2.set_xlabel(x_var)
    ax2.set_ylabel("Mean Width")
   
    # Adjust layout to prevent overlapping
    fig.tight_layout()

    plt.show()


def plot_box_for_specific_function(df,x_var, y_var, hue, y2_var, filter_function, x_axis_name, y_var_title1,y_var_title2,x2_var):
    df = df[df["data_name"] == filter_function]
    
    specific_order_layer = ['(200, 1000)', '(400, 2000)', '(200, 1000, 1000, 1000)', '(400, 2000, 2000, 2000)']
    if x_var == "layer_size":
        df = df.copy()
        df[x_var] = pd.Categorical(df[x_var], categories=specific_order_layer, ordered=True)
        df.sort_values(by=x_var, inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
   
    # First boxplot
    sns.boxplot(data=df, x=x_var, y=y_var, hue=hue, ax=ax1)
    if y_var_title1 =="Coverage":
        ax1.axhline(1-alpha, ls="--", color="k")
    
        
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_var_title1)
   
    # Second boxplot
    sns.boxplot(data=df, x=x2_var, y=y2_var, hue=hue, ax=ax2)

    ax2.set_xlabel(x_axis_name)
    ax2.set_ylabel(y_var_title2)
    
    if x_var == "layer_size":
        ax1.tick_params(axis='x', rotation=15)
        ax2.tick_params(axis='x',  rotation=15)
    fig.suptitle(f"Training data size against mean width for {filter_function} function")
    # Adjust layout to prevent overlapping
    fig.tight_layout()

    plt.show()
   
def plot_box_for_specific_function_and_noise_type(df,x_var, y_var, hue, y2_var, filter_function, x_axis_name,noise_type, y_var_title1,y_var_title2):
    df = df.copy()

    df = df[df["data_name"] == filter_function]
    df = df[df["noise_type"] == noise_type]
    if noise_type == "student":
        df[x_var] = pd.Categorical(df[x_var], categories=sorted(df[x_var].unique(), reverse=True), ordered=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Noise scaling factor for {filter_function} function and {noise_type} noise")
    # First boxplot
    sns.boxplot(data=df, x=x_var, y=y_var, hue=hue, ax=ax1)
    if y_var_title1 =="Coverage":
        ax1.axhline(1-alpha, ls="--", color="k")
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_var_title1)
   
    # Second boxplot
    sns.boxplot(data=df, x=x_var, y=y2_var, hue=hue, ax=ax2)

    ax2.set_xlabel(x_axis_name)
    ax2.set_ylabel(y_var_title2)

    # Adjust layout to prevent overlapping
    fig.tight_layout()
    # if noise_type == "student":
    #     ax1.set_xticklabels(ax1.get_xticklabels()[::-1])
    #     ax2.set_xticklabels(ax2.get_xticklabels()[::-1])

    plt.show()    
   
def get_summary(x_var_array,x_var_label_array, filter_function="cube",y_var="coverage", y2_var="mean_width", y_var_title1="Coverage",y_var_title2="Mean Width"):
   
   
    for x_var, x_var_label in zip(x_var_array,x_var_label_array):
        plot_box_for_specific_function(df,x_var=x_var, y_var=y_var, hue=None, y2_var=y2_var, filter_function=filter_function, x_axis_name=x_var_label, y_var_title1=y_var_title1,y_var_title2=y_var_title2,x2_var=x_var)
    plot_box_for_specific_function_and_noise_type(df,x_var="noise_std", y_var=y_var, hue=None, y2_var=y2_var, filter_function=filter_function, x_axis_name="Noise scaling factor",noise_type="student", y_var_title1=y_var_title1,y_var_title2=y_var_title2,)
    plot_box_for_specific_function_and_noise_type(df,x_var="noise_std", y_var=y_var, hue=None, y2_var=y2_var, filter_function=filter_function, x_axis_name="Noise scaling factor",noise_type="normal", y_var_title1=y_var_title1,y_var_title2=y_var_title2,)

def preprocess_ssc(ssc_str):
    # Remove multiple spaces and replace with a single space
    ssc_str = ' '.join(ssc_str.split())
    # Replace spaces between numbers with commas
    ssc_str = ssc_str.replace(' ', ',')
    return ssc_str

def plot_ssc(df, strategy, data_name, noise_std, num_samples, noise_type, layer_size):
    # Filter the DataFrame based on the parameters
    filtered_data = df[
        (df['strategy'] == strategy) &
        (df['data_name'] == data_name) &
        (df['noise_std'] == noise_std) &
        (df['num_samples'] == num_samples) &
        (df['noise_type'] == noise_type) &
        (df['layer_size'] == layer_size)
    ]
    
    # Extract and process 'ssc' data
    if not filtered_data.empty:
        ssc_str = filtered_data['ssc'].values[0]
        ssc_str = preprocess_ssc(ssc_str)
        ssc_list = ast.literal_eval(ssc_str)[0]
        x_labels = range(len(ssc_list))

        # Plotting
        plt.bar(x_labels, ssc_list)
        plt.axhline(1-alpha, ls="--", color="k")
        plt.xlabel('Index of increasing PI length')
        plt.ylabel('SSC Value')
        plt.title(f'Coverage for varying PI length for {strategy}, {data_name}, {noise_std}, {num_samples}, {noise_type}, {layer_size}')
        plt.show()
    else:
        print("No data found for the specified parameters.")  


#%%
THING_TO_ITERATE_OVER = "strategy"
   
   
#strategy, data_name, noise_type, layer_size, num_samples, noise_std ="jackknife_plus_ab","linear","normal","(200, 1000)",1000,1
x_var = THING_TO_ITERATE_OVER
plot_coverage_for_specific(df, strategy="jackknife_plus_ab", data_name="cube", noise_type="normal", layer_size="(200, 1000)", num_samples=1000, noise_std=1, x_var="strategy")
plot_coverage_for_specific(df, strategy="jackknife_plus_ab", data_name="linear", noise_type="normal", layer_size="(200, 1000)", num_samples=1000, noise_std=1, x_var="strategy")


# plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1, x_var="strategy",
#                                strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="student", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=4, x_lab="Strategy")


# plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1, x_var="noise_std",
#                                strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="student", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=4, x_lab="Noise Scaling Factor")

plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=100, noise_std1=1, x_var="layer_size",
                               strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="normal", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=1, x_lab="Layer Size")

plot_coverage_for_specific_multi(df, strategy1="jackknife_plus_ab", data_name1="cube", noise_type1="normal", layer_size1="(200, 1000)", num_samples1=1000, noise_std1=1, x_var="layer_size",
                               strategy2="jackknife_plus_ab", data_name2="cube", noise_type2="student", layer_size2="(200, 1000)", num_samples2=1000, noise_std2=4, x_lab="layer_size")
#%%

base_path = 'results'
function_name = "linear"
noise_type = "normal"
strategy = "jackknife_plus_ab"
num_samples_str = "1000"
std = "0.5"

iterated_parameter = THING_TO_ITERATE_OVER
parameter_list = ["jackknife_plus","split","cqr","jackknife_plus_ab"]
parameter_list_strategy = ["jackknife_plus","split","cqr","jackknife_plus_ab"]
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
  
plotter = Plotter(base_path)
#plotter.plot_images(function_name, num_samples_str, noise_type, std, strategy, iterated_parameter,parameter_list)

# plotter.plot_images(function_name="linear", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
# plotter.plot_images(function_name="cube", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
# plotter.plot_images(function_name="sinex_het", num_samples_str="300", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)
plotter.plot_images(function_name="nmm", num_samples_str="1000", noise_type="normal", std="0.5", strategy="cqr",layer_size="(200, 1000)", iterated_parameter="strategy",parameter_list=parameter_list_strategy)

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
# get_summary(x_var_array,x_var_label_array, filter_function="cube",y_var="coverage", y2_var="mean_width")
get_summary(x_var_array,x_var_label_array, filter_function="cube",y_var="coverage", y2_var="mean_width")
get_summary(x_var_array,x_var_label_array, filter_function="cube",y_var="mwi", y2_var="cwc", y_var_title1="Mean Winkler Score",y_var_title2="Coverage-Width Based Criterion")
# get_summary(x_var_array,x_var_label_array, filter_function="cube",y_var="mwi", y2_var="cwc", y_var_title1="Mean Winkler Score",y_var_title2="Coverage-Width Based Criterion")
# get_summary(x_var_array,x_var_label_array, filter_function="sinex_het",y_var="coverage", y2_var="mean_width", y_var_title1="Coverage",y_var_title2="Mean Width")
# get_summary(x_var_array,x_var_label_array, filter_function="sinex_het",y_var="mwi", y2_var="cwc", y_var_title1="Mean Winkler Score",y_var_title2="Coverage-Width Based Criterion")



#%%
test_titles = ["1","2","3","4"]
plotter.plot_images_from_dir('test',test_titles)


#PLOT SIZE STRATIFIED COVERAGE
plot_ssc(df, 'jackknife_plus', 'nmm', 0, 1000, 'normal', '(200, 1000)')

#plot_box_for_specific_function(df,x_var="", y_var=y_var, hue=None, y2_var=y2_var, filter_function=filter_function, x_axis_name=x_var_label, y_var_title1=y_var_title1,y_var_title2=y_var_title2,x2_var=x_var)
# noise_type = "normal"
# stds = [0,0.2,0.5,1] [[1.         0.95       1.         1.         0.53333333]]

# coverage_for_each_std(file_name, noise_type, num_samples)
# coverage_for_each_strategy(file_name, stds, noise_type, num_samples)
# [strategy,      data_name   ,coverage   ,mean_width ,cwc  ,mwi  ,total_time ,noise_std  ,num_samples      ,noise_type ,layer_size ,ssc]
#[strategy,      data_name  , noise_std  ,num_samples      ,noise_type ,layer_size ,ssc]