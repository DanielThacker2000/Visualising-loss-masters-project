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

rng = np.random.RandomState(42) #Set seed
alpha = 0.1
#%%
# =============================================================================
# Plotting
# =============================================================================

def plot_ci(x_test, y_test, y_pred, y_ci, coverage, mean_width, name, conformal_method):
    # for val in range(len(y_ci[:, 0, 0])):
    #     print(y_ci[val, 0, 0]-y_ci[val, 1, 0])
    order = np.argsort(x_test.flatten())#set order for coordinates
    plt.figure(figsize=(8, 8))
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
    plt.scatter(x_test, y_test, color="red", alpha=0.7, label="testing", s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{conformal_method} confidence intervals for MLP with {name} function with total coverage {coverage:.2f} and mean width {mean_width:.2f}")
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
        ncol=3
    )
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

def coverage_for_each_strategy(file_name, stds):
    data = pd.read_csv(file_name)

    # Set the width of the bars
    bar_width = 0.4
    for std in stds:
        # Iterate over each unique 'data_name'
        for data_name in data['data_name'].unique():
            # Filter the data for the current 'data_name'
            df = data[(data['data_name'] == data_name) & (data['noise_std'] == std)]
            
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
            ax1.set_xticklabels(df['strategy'])
            
            ax1.set_ylim(0, df['coverage'].max() * 1.1)
            ax2.set_ylim(0, df['mean_width'].max() * 1.1)
            
            # Add title and legend
            plt.title(f'Coverage and Mean Width for {data_name} function with noise standard deviation {std}')
            ax1.axhline(1-alpha, ls="--", color="k")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Show the plot
            plt.show()
            
def coverage_for_each_std(file_name):
    data = pd.read_csv(file_name)

    # Set the width of the bars
    bar_width = 0.4
    for stategy in data['strategy'].unique():
    # Iterate over each unique 'data_name'
        for data_name in data['data_name'].unique():
            # Filter the data for the current 'data_name'
            df = data[(data['data_name'] == data_name) & (data['strategy'] == stategy)]
            
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
            ax1.set_xticklabels(df['noise_std'])
            
            ax1.set_ylim(0, df['coverage'].max() * 1.1)
            ax2.set_ylim(0, df['mean_width'].max() * 1.1)
            
            # Add title and legend
            plt.title(f'Coverage and Mean Width for {data_name} function and {stategy} strategy')
            ax1.axhline(1-alpha, ls="--", color="k")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Show the plot
            plt.show()