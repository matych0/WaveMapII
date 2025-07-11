#%%
import optuna
from optuna.visualization import matplotlib as vis_matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
optuna.visualization.plot_contour(study, params=['learning_rate', 'batch_size', 'n_controls'], target_name="Concordance Index")

#%%

log_params = [0,2]

fig = vis_matplotlib.plot_contour(study, params=['cox_regularization', 'weight_decay', 'dropout'], target_name="Concordance Index")

fig = plt.gcf()
ax = np.array(fig.axes)[:-1].reshape(3, 3)

fig.suptitle("")

ax[0][0].set_facecolor('white') 
ax[1][1].set_facecolor('white')
ax[2][2].set_facecolor('white')

#ax.set_title("")

for i, ax_row in enumerate(ax):
    for j, _ in enumerate(ax_row):
        ax[i][j].set_aspect('auto')
        
        #ax_col.tick_params(axis='both', colors='black', labelsize=10)
        if i < len(ax) - 1:
            ax[i][j].set_xlabel('')
            ax[i][j].set_xticks([])
        if j > 0:
            ax[i][j].set_ylabel('')
            ax[i][j].set_yticks([])
            
        if i in log_params:
            if j in log_params:
                ax[i][j].set_xlim(left=1e-4, right=1e-1)
                ax[i][j].set_xscale('log')
            
                ax[i][j].set_ylim(bottom=1e-4, top=1e-1)
                ax[i][j].set_yscale('log')
        """ ax_col.set_xlabel(ax_col.get_xlabel(), fontsize=10, color='black')
        ax_col.set_ylabel(ax_col.get_ylabel(), fontsize=10, color='black')
        for label in ax_col.get_xticklabels() + ax_col.get_yticklabels():
            label.set_color('black')
            label.set_fontsize(10) """
""" ax.set_xscale('log')
ax.set_xlabel('Log Param1', fontsize=12, color='black')
ax.set_ylabel('Param2', fontsize=12, color='black')
ax.tick_params(axis='both', colors='black', labelsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False) """

""" colorbar_ax = fig.axes[-1]
colorbar_ax.set_facecolor('white')
colorbar_ax.tick_params(colors='black') """
#%%
    



    
if __name__ == "__main__":
    plot_optuna_history(study)
