# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.axisbelow'] = True 


r2_data = {
    'DNN':      [0.985, 0.945, 0.933],  
    'LightGBM': [0.979, 0.977, 0.972],
    'XGBoost':  [0.994, 0.981, 0.972],
    'Stacking': [0.999, 0.976, 0.971],
    'Voting':   [0.989, 0.985, 0.957]
}

mape_data = {
    'DNN':      [16.23, 20.77, 28.44],    
    'LightGBM': [9.76, 9.85, 10.32],
    'XGBoost':  [3.75, 9.28, 8.82],
    'Stacking': [1.57, 8.45, 9.39],
    'Voting':   [6.81, 7.51, 12.54]
}


categories = ['Train', 'Test', 'Unseen']

colors = {
    'DNN':      '#E74C3C', 
    'LightGBM': '#2ECC71', 
    'XGBoost':  '#3498DB',
    'Stacking': '#F1C40F', 
    'Voting':   '#9B59B6'  
}

def plot_comparison(data_dict, title, ylabel, is_r2=True):
    models = list(data_dict.keys())
    values = np.array(list(data_dict.values()))
    
    n_groups = len(categories)
    n_models = len(models)
    
    bar_width = 0.15
    indices = np.arange(n_groups)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, model in enumerate(models):
        pos = indices + (i - n_models/2 + 0.5) * bar_width
        val = values[i]
        
        bars = ax.bar(pos, val, bar_width, label=model, color=colors[model], edgecolor='black', alpha=0.9, zorder=3)
        
        for bar in bars:
            height = bar.get_height()
            offset = 0.01 if is_r2 else 0.1
            rotation = 90
            
            txt = f'{height:.3f}' if is_r2 else f'{height:.2f}%'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    txt,
                    ha='center', va='bottom', fontsize=10, rotation=rotation, fontweight='bold', color='black')

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=12)
    
    if is_r2:
        min_val = np.min(values)
        ax.set_ylim(min_val - 0.05, 1.05) 
    else:
        max_val = np.max(values)
        ax.set_ylim(0, max_val * 1.2)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(direction='in', top=False, right=False)
    
    plt.tight_layout()
    plt.show()


plot_comparison(r2_data, 'Model Performance Comparison ($R^2$ Score)', '$R^2$ Value', is_r2=True)

plot_comparison(mape_data, 'Model Error Comparison (MAPE)', 'MAPE (%)', is_r2=False)
