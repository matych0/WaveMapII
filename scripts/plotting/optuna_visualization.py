#%%
import optuna
from optuna.visualization import matplotlib as vis_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna.visualization as vis
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.io import show
import plotly.express as px


#pio.renderers.default = "browser"
#%%
study = optuna.load_study(study_name='hyperparam_optim_mult_controls_CUDA', storage='sqlite:///D:/logs/logs_160_runs/hyperparam_optim_mult_controls_CUDA')

#%%
def plot_optuna_history(study):
    ax = vis_matplotlib.plot_optimization_history(study, target_name="Trial Concordance Index")
    fig = ax.figure

    ax.legend(loc='lower right', frameon=True, fontsize=10)


    # Background color
    ax.set_facecolor('white')               # Plot background
    fig.patch.set_facecolor('white')        # Figure background

    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    ax.set_title('')
    ax.tick_params(axis='both', colors='black', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(10)
        
    ax.set_xlabel('Trial', fontsize=10, color='black')
    ax.set_ylabel('Concordance Index', fontsize=10, color='black')
        
    # Style for the best-so-far line
    lines = ax.get_lines()
    best_line = lines[0]
    best_line.set_color('darkred')
    best_line.set_linewidth(2.0)
    best_line.set_linestyle('-')                # Solid line 

    ax.set_ylim(0.5, 0.65)  # Adjust y-axis limits as needed

    plt.tight_layout()
    plt.show()


#%%
def plot_optuna_contour(study, params=['learning_rate', 'batch_size', 'n_controls'], target_name="Concordance Index"):
    fig = optuna.visualization.plot_contour(study, params=params, target_name=target_name)

    pio.write_image(fig, "C:/Users/marti/Documents/Diplomka/images/optuna_contour_plot_training.pdf",  width=600, height=600)

    show(fig)


#%%
def plot_optuna_param_importances(study):
    ax = vis_matplotlib.plot_param_importances(study, target_name="Concordance Index")
    fig = ax.figure

    ax.get_legend().remove()
    ax.set_title("")
    # Background color
    ax.set_facecolor('white')               # Plot background
    fig.patch.set_facecolor('white')

    ax.set_title('')
    ax.tick_params(axis='both', colors='black', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(10)
        
    ax.set_xlabel('Hyperparameter Importance', fontsize=11, color='black')
    ax.set_ylabel('Hyperparameter', fontsize=11, color='black')
        
    ax.grid(True, linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    ax.figure.tight_layout()
    
    plt.show()


#%%
#vis_matplotlib.plot_slice(study, params=["num_epochs"], target_name="Concordance Index")

def plot_optuna_slice(study):
    ax = vis_matplotlib.plot_slice(study, params=["num_epochs"], target_name="Concordance Index")
    fig = ax.figure
    
    ax.set_title("")
    # Background color
    ax.set_facecolor('white')               # Plot background
    fig.patch.set_facecolor('white')

    ax.set_title('')
    ax.tick_params(axis='both', colors='black', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(10)

    ax.set_xlabel('Epochs', fontsize=11, color='black')
    ax.set_ylabel('Concordance Index', fontsize=11, color='black')
        
    ax.grid(True, linestyle='--', alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    ax.set_ylim(0.5, 0.65)  # Adjust y-axis limits as needed
    ax.set_xscale('log')
    ax.set_xticks([50,60,70,80,90, 100, 200, 300, 400, 500])
    ax.set_xticklabels([50,60,70,80,90, 100, 200, 300, 400, 500])  # Set x-ticks from 50 to 500 with a step of 10

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    ax.figure.tight_layout()
    
    plt.show()

#%%

    
if __name__ == "__main__":
    #plot_optuna_history(study)
    %matplotlib qt
    #plot_optuna_param_importances(study)
    plot_optuna_slice(study)
