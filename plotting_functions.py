# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:08:35 2024

@author: dan
"""
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.metrics import (regression_coverage_score, regression_mean_width_score, regression_ssc,regression_ssc_score)
from mapie.subsample import Subsample
from mapie.conformity_scores import GammaConformityScore,ResidualNormalisedScore
import numpy as np
import pandas as pd
import os
from PIL import Image
rng = np.random.RandomState(42) #Set seed
alpha = 0.1

class Plotter:
    def __init__(self, base_path):
        self.base_path = base_path
    
    def load_images(self, function_name,num_samples, noise_type, std_list, strategy):
        """
        Load images from the specified directories.
        
        Parameters:
        - function_name (str): The name of the function.
        - noise_type (str): The type of noise.
        - std_list (list): A list of standard deviations.
        - strategy (str): The strategy name.
        
        Returns:
        - images (list): A list of tuples containing the std value and the corresponding image.
        """
        images = []
        for std in std_list:
            std_str = str(std)
            dir_path = os.path.join(self.base_path, function_name, num_samples, noise_type, std_str, strategy)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".png"):
                        img_path = os.path.join(dir_path, file)
                        img = Image.open(img_path)
                        images.append((std, img))
            else:
                print(f"Directory does not exist: {dir_path}")
        return images
    
    def plot_images(self, function_name, num_samples, noise_type, std_list, strategy):
        """
        Plot the loaded images in a formatted subplot.
        
        Parameters:
        - function_name (str): The name of the function.
        - noise_type (str): The type of noise.
        - std_list (list): A list of standard deviations.
        - strategy (str): The strategy name.
        """
        images = self.load_images(function_name,num_samples, noise_type, std_list, strategy)
        
        if not images:
            print("No images to plot.")
            return
        
        num_images = len(images)
        cols =2  # Number of columns in the subplot
        rows = (num_images // cols) + (num_images % cols > 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(24, 12 * rows))
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
        
        for ax in axes:
            ax.axis('off')  # Hide the axes by default
        
        for idx, (std, img) in enumerate(images):
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(f"std: {std}")
            ax.axis('off')  # Show the axes
        
        plt.suptitle(f"{function_name} - {noise_type} - {strategy}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0, h_pad=0, w_pad=0)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.show()

def create_save_directory(graph_name, function_name,num_samples, noise_type, std, strategy, base_path):

    """
    Create a directory path based on the given parameters and create the directory if it does not exist.
    
    Parameters:
    - base_path (str): The base path where the directories should be created.
    - function_name (str): The name of the function.
    - noise_type (str): The type of noise.
    - std (float or str): The standard deviation.
    - strategy (str): The strategy name.
    
    Returns:
    - save_path (str): The full path of the created directory.
    """
    # Convert std to string if it is not already
    std_str = str(std)
    
    # Construct the directory path
    save_path = os.path.join(base_path, function_name,num_samples, noise_type, std_str, strategy)
    
    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory created: {save_path}")
        
    save_path = os.path.join(save_path, graph_name)
    return save_path

#%%
# =============================================================================
# Plotting
# =============================================================================

def plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, name, conformal_method, noise_type,std,num_samples):
    # for val in range(len(y_ci[:, 0, 0])):
    #     print(y_ci[val, 0, 0]-y_ci[val, 1, 0])
    base_path = "results"
    save_path = create_save_directory("PIs.png", name,num_samples, noise_type, std, conformal_method, base_path)
    order = np.argsort(x_test.flatten())#set order for coordinates
    plt.figure(figsize=(12, 12))
    plt.plot(
        x_test[order],
        y_pred[order],
        label="Predictions",
        color="green"
    )
    plt.fill_between(
        x_test.flatten()[order],
        y_ci[:, 0, 0][order],
        y_ci[:, 1, 0][order],
        alpha=0.4,
        label="Confidence Intervals",
        color="green"
    )
    plt.scatter(x_test, y_test, color="red", alpha=0.7, label="testing", s=7)
    plt.xlabel("x", fontsize=25)
    plt.ylabel("y", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title(
    #     f"{conformal_method} confidence intervals for MLP with {name} function with total coverage {coverage:.2f} and mean width {mean_width:.2f}",
    #           fontsize=20)
    # plt.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.07),
    #     fancybox=True,
    #     shadow=False,
    #     ncol=3,
    #     fontsize=16
    # )
    plt.savefig(save_path)
    plt.show()

#Returns data coverage within specific x intervals for plotting
def get_coverage(y_test, y_cis, STRATEGIES, num_bins, x_test, func_name,std):
    recap = {}
    bins = np.linspace(min(x_test),max(x_test),num_bins)
    bin_labels = []
    for i in range(len(bins)-1):
        bin1, bin2 = bins[i], bins[i+1]
        bin1lab, bin2lab = float(bins[i]), float(bins[i + 1])
        bin_labels.append(f"[{bin1lab:.2f}, {bin2lab:.2f}]")
        name = f"[{bin1}, {bin2}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where((x_test >= bins[i]) * (x_test < bins[i+1]))
            y_test_trunc = np.take(y_test[func_name][std], indices)
            y_low_ = np.take(y_cis[strategy][func_name][std][:, 0, 0], indices)
            y_high_ = np.take(y_cis[strategy][func_name][std][:, 1, 0], indices)
            score_coverage = regression_coverage_score(
                y_test_trunc[0], y_low_[0], y_high_[0]
            )
            recap[name].append(score_coverage)
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    recap_df.T.plot.bar(figsize=(12, 5), alpha=0.7)
    plt.axhline(1-alpha, ls="--", color="k")
    plt.ylabel("Conditional Coverage")
    plt.xlabel("X bins")
    plt.xticks(ticks=np.arange(len(bin_labels)), labels=bin_labels, rotation=45, ha='right')
    plt.ylim(0.6, 1.0)
    plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.title(f"Coverage for each conformal strategy with {func_name} noise function for noise std {std}")
    plt.show()
    return recap_df


def get_heteroscedastic_coverage(y_test, y_cis, STRATEGIES, bins, x_test, func_name):
    recap = {}
    for i in range(len(bins)-1):
        bin1, bin2 = bins[i], bins[i+1]
        name = f"[{bin1}, {bin2}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where((x_test[func_name] >= bins[i]) * (x_test[func_name] <= bins[i+1]))
            y_test_trunc = np.take(y_test[func_name], indices)
            y_low_ = np.take(y_cis[strategy][func_name][:, 0, 0], indices)
            y_high_ = np.take(y_cis[strategy][func_name][:, 1, 0], indices)
            score_coverage = regression_coverage_score(
                y_test_trunc[0], y_low_[0], y_high_[0]
            )
            recap[name].append(score_coverage)
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    recap_df.T.plot.bar(figsize=(12, 5), alpha=0.7)
    plt.axhline(1-alpha, ls="--", color="k")
    plt.ylabel("Conditional coverage")
    plt.xlabel("x bins")
    plt.xticks(rotation=0)
    plt.ylim(0.8, 1.0)
    plt.legend(fontsize=8, loc="upper right")
    plt.title("Coverage for each conformal strategy with heteroscedastic noise function")
    plt.show()
    return recap_df

def plot_widths_along_x(training_data_name, STRATEGIES, x_test,y_cis, stds):

    for name in training_data_name:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for strategy in STRATEGIES.keys():
            for std in stds:
                
                ax.plot(
                    x_test[name][std],
                    y_cis[strategy][name][std][:, 1, 0].ravel() - y_cis[strategy][name][std][:, 0, 0].ravel(),
                    label=strategy
                )
        ax.set_xlabel("x")
        ax.set_ylabel("Confidence Interval Width")
        ax.legend(fontsize=8)
        ax.set_title(f"Confidence interval widths for {name} function")
        base_path = "widths_along_x_ims"
        #save_path = create_save_directory("widths_along_x.png", name,num_samples, noise_type, std, conformal_method, base_path)
        plt.show()
       
def plot_ci_real_data(method,y_test_sorted,y_pred_sorted,lower_bound,upper_bound,coverage,width, title):

    lower_bound_ = lower_bound
    y_pred_sorted_ = y_pred_sorted
    y_test_sorted_ = y_test_sorted
   
    # lower_bound_ = lower_bound_[0:np.round(len(lower_bound_)/3)]
    # y_pred_sorted_ = y_pred_sorted_[0:np.round(len(y_pred_sorted_)/3)]
    # y_test_sorted_ = y_test_sorted_[0:np.round(len(y_test_sorted_)/3)]
   
    error = y_pred_sorted_-lower_bound_
    fig, axs = plt.subplots(1,1)
    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=np.abs(error[~warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        label="Inside prediction interval"
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=np.abs(error[warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red",
        label="Outside prediction interval"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*", color="green",
        label="True value"
    )
    axs.set_xlabel("True target value")
    axs.set_ylabel("Predicted target value")
    # ab = AnnotationBbox(
    #     TextArea(
    #         f"Coverage: {coverage:.2f}\n"
    #         + f"Interval width: {width:.2f}"
    #     ),
    #     xy=(0, 1),
    #     )
    # lims = [
    #     np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
    #     np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    # ]
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    min_val = np.min([xlim, ylim])
    max_val = np.max([xlim, ylim])
    x_vals = np.linspace(max_val, 0, 500)
    y_vals = x_vals
    axs.plot(x_vals, y_vals, '--', alpha=0.75, color="black", label="x=y")
    # axs.add_artist(ab)
    axs.set_title(f"Predicted values using the {method} method on {title}. Coverage {coverage:.2f}, MPIW {width:.2f}", fontweight='bold')

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
            
            ax1.set_ylim(0, df['coverage'].max() * 1.1)
            ax2.set_ylim(0, df['mean_width'].max() * 1.1)
            
            # Add title and legend
            plt.title(f'Coverage and Mean Width for {data_name} function with noise standard deviation {std}')
            ax1.axhline(1-alpha, ls="--", color="k")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            base_path = "results_strategy_coverage"
            save_path = create_save_directory("coverage_against_strategy.png", data_name,num_samples, noise_type, std, "pointless",base_path)
            plt.savefig(save_path)
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
            
            save_path = create_save_directory("coverage_against_std.png", data_name,num_samples, noise_type, "pointless", stategy,base_path)
            plt.savefig(save_path)
            # Show the plot
            plt.show()