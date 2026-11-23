#!/usr/bin/env python3
"""
Comprehensive visualization script
Combines all chart generation functions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import torch
from functions import VGGA, VGG_A_BatchNorm

# Set fonts that support Chinese characters
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_training_comparison():
    """Plot the training curves of the standard comparison experiment"""
    print("Generating the training comparison plot...")
    
    # Load the data
    with open('results/standard_vgg_results.json', 'r') as f:
        std_results = json.load(f)
    with open('results/vgg_batchnorm_results.json', 'r') as f:
        bn_results = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loss curves
    ax1.plot(std_results['train_losses'], label='Standard VGG', linewidth=2)
    ax1.plot(bn_results['train_losses'], label='BatchNorm VGG', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy curves
    ax2.plot(std_results['val_accs'], label='Standard VGG', linewidth=2)
    ax2.plot(bn_results['val_accs'], label='BatchNorm VGG', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multi_lr_results():
    """Plot the results of the multi-learning-rate experiment"""
    print("Generating the multi-learning-rate comparison plot...")
    
    lr_dir = Path('results/multi_lr_experiment')
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        
        # Load the standard VGG results
        std_file = lr_dir / f'standard_vgg_lr_{lr}_results.json'
        if std_file.exists():
            with open(std_file, 'r') as f:
                std_data = json.load(f)
                ax.plot(std_data['train_losses'], label='Standard VGG', linewidth=2)
        
        # Load the BatchNorm VGG results
        bn_file = lr_dir / f'vgg_batchnorm_lr_{lr}_results.json'
        if bn_file.exists():
            with open(bn_file, 'r') as f:
                bn_data = json.load(f)
                ax.plot(bn_data['train_losses'], label='BatchNorm VGG', linewidth=2)
        
        ax.set_title(f'Learning rate = {lr}', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Training Loss', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3)
    
    plt.suptitle('Training Loss Comparison Across Learning Rates', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/multi_lr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_gradient_analysis():
    """Plot the gradient analysis"""
    print("Generating the gradient analysis plot...")
    
    lr_dir = Path('results/multi_lr_experiment')
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gradient predictiveness
    for model_type in ['standard_vgg', 'vgg_batchnorm']:
        predictiveness = []
        for lr in learning_rates:
            result_file = lr_dir / f'{model_type}_lr_{lr}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'gradient_predictiveness' in data:
                        # Take the mean
                        avg_pred = np.mean([p for p in data['gradient_predictiveness'] if p is not None])
                        predictiveness.append(avg_pred)
                    else:
                        predictiveness.append(0)
        
        label = 'BatchNorm VGG' if 'batchnorm' in model_type else 'Standard VGG'
        ax1.plot(range(len(learning_rates)), predictiveness, 'o-', label=label, linewidth=2, markersize=8)
    
    ax1.set_xticks(range(len(learning_rates)))
    ax1.set_xticklabels([f'{lr}' for lr in learning_rates])
    ax1.set_xlabel('Learning Rate', fontsize=12)
    ax1.set_ylabel('Gradient Predictiveness (Cosine Similarity)', fontsize=12)
    ax1.set_title('Gradient Predictiveness Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient variance
    for model_type in ['standard_vgg', 'vgg_batchnorm']:
        variances = []
        for lr in learning_rates:
            result_file = lr_dir / f'{model_type}_lr_{lr}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'gradient_variance' in data:
                        # Take the mean of the log values
                        avg_var = np.mean([np.log10(v) for v in data['gradient_variance'] if v > 0])
                        variances.append(avg_var)
                    else:
                        variances.append(-8)
        
        label = 'BatchNorm VGG' if 'batchnorm' in model_type else 'Standard VGG'
        ax2.plot(range(len(learning_rates)), variances, 'o-', label=label, linewidth=2, markersize=8)
    
    ax2.set_xticks(range(len(learning_rates)))
    ax2.set_xticklabels([f'{lr}' for lr in learning_rates])
    ax2.set_xlabel('Learning Rate', fontsize=12)
    ax2.set_ylabel('Gradient Variance (log10)', fontsize=12)
    ax2.set_title('Gradient Variance Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/gradient_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_filter_visualization():
    """Visualize the first-layer convolution filters"""
    print("Generating the filter visualization...")
    
    device = torch.device('cpu')
    
    # Load the models
    std_model = VGGA(use_batchnorm=False)
    bn_model = VGG_A_BatchNorm()
    
    # Try to load the trained weights
    std_checkpoint = 'results/checkpoints/standard_vgg_checkpoint.pt'
    bn_checkpoint = 'results/checkpoints/vgg_batchnorm_checkpoint.pt'
    
    if Path(std_checkpoint).exists():
        checkpoint = torch.load(std_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            std_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            std_model.load_state_dict(checkpoint)
    
    if Path(bn_checkpoint).exists():
        checkpoint = torch.load(bn_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            bn_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            bn_model.load_state_dict(checkpoint)
    
    # Get the first convolution layer weights
    std_filters = std_model.features[0].weight.data.cpu().numpy()
    bn_filters = bn_model.features[0].weight.data.cpu().numpy()
    
    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standard VGG filters
    n_filters = min(64, std_filters.shape[0])
    grid_size = int(np.sqrt(n_filters))
    
    for i in range(n_filters):
        ax1.subplot(grid_size, grid_size, i+1)
        filter_img = std_filters[i].transpose(1, 2, 0)
        # Normalize to [0, 1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        ax1.imshow(filter_img)
        ax1.axis('off')
    
    ax1.set_title('Standard VGG First-Layer Filters', fontsize=14)
    
    # BatchNorm VGG filters
    for i in range(n_filters):
        ax2.subplot(grid_size, grid_size, i+1)
        filter_img = bn_filters[i].transpose(1, 2, 0)
        # Normalize to [0, 1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        ax2.imshow(filter_img)
        ax2.axis('off')
    
    ax2.set_title('BatchNorm VGG First-Layer Filters', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/filter_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimization_results():
    """Plot the results of the optimization strategy experiments"""
    print("Generating the optimization strategy comparison plot...")
    
    opt_dir = Path('results/optimization_experiments')
    
    # Check whether the result files exist
    if not opt_dir.exists():
        print("Optimization strategy results not found, skipping...")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Activation function comparison
    if (opt_dir / 'activation_results.json').exists():
        with open(opt_dir / 'activation_results.json', 'r') as f:
            act_results = json.load(f)
        
        activations = ['relu', 'leaky_relu', 'elu', 'swish']
        std_errors = []
        bn_errors = []
        
        for act in activations:
            if act in act_results.get('Standard VGG', {}):
                error = act_results['Standard VGG'][act].get('best_test_error', 100)
                std_errors.append(error if error != 'N/A' else 100)
            else:
                std_errors.append(100)
                
            if act in act_results.get('BatchNorm VGG', {}):
                error = act_results['BatchNorm VGG'][act].get('best_test_error', 100)
                bn_errors.append(error if error != 'N/A' else 100)
            else:
                bn_errors.append(100)
        
        x = np.arange(len(activations))
        width = 0.35
        
        ax1.bar(x - width/2, std_errors, width, label='Standard VGG')
        ax1.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax1.set_xlabel('Activation Function')
        ax1.set_ylabel('Test Error (%)')
        ax1.set_title('Test Error by Activation Function')
        ax1.set_xticks(x)
        ax1.set_xticklabels(activations)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Loss function comparison
    if (opt_dir / 'loss_function_results.json').exists():
        with open(opt_dir / 'loss_function_results.json', 'r') as f:
            loss_results = json.load(f)
        
        loss_names = ['cross_entropy', 'cross_entropy_l1', 'cross_entropy_l2', 'label_smoothing']
        loss_labels = ['CE', 'CE+L1', 'CE+L2', 'LS']
        std_errors = []
        bn_errors = []
        
        for loss in loss_names:
            if loss in loss_results.get('Standard VGG', {}):
                std_errors.append(loss_results['Standard VGG'][loss].get('best_test_error', 100))
            else:
                std_errors.append(100)
                
            if loss in loss_results.get('BatchNorm VGG', {}):
                bn_errors.append(loss_results['BatchNorm VGG'][loss].get('best_test_error', 100))
            else:
                bn_errors.append(100)
        
        x = np.arange(len(loss_names))
        
        ax2.bar(x - width/2, std_errors, width, label='Standard VGG')
        ax2.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax2.set_xlabel('Loss Function')
        ax2.set_ylabel('Test Error (%)')
        ax2.set_title('Test Error by Loss Function')
        ax2.set_xticks(x)
        ax2.set_xticklabels(loss_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Optimizer comparison
    if (opt_dir / 'optimizer_results.json').exists():
        with open(opt_dir / 'optimizer_results.json', 'r') as f:
            opt_results = json.load(f)
        
        optimizers = ['SGD', 'SGD+Momentum', 'Adam', 'AdamW', 'RMSprop']
        std_errors = []
        bn_errors = []
        
        for opt in optimizers:
            if opt in opt_results.get('Standard VGG', {}):
                std_errors.append(opt_results['Standard VGG'][opt].get('best_test_error', 100))
            else:
                std_errors.append(100)
                
            if opt in opt_results.get('BatchNorm VGG', {}):
                bn_errors.append(opt_results['BatchNorm VGG'][opt].get('best_test_error', 100))
            else:
                bn_errors.append(100)
        
        x = np.arange(len(optimizers))
        
        ax3.bar(x - width/2, std_errors, width, label='Standard VGG')
        ax3.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax3.set_xlabel('Optimizer')
        ax3.set_ylabel('Test Error (%)')
        ax3.set_title('Test Error by Optimizer')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimizers, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table():
    """Generate the experiment results summary table"""
    print("Generating the results summary table...")
    
    # Collect all results
    summary = {
        'Standard Comparison': {},
        'Multiple Learning Rates': {},
        'Optimization Strategies': {}
    }
    
    # Standard comparison results
    if Path('results/standard_vgg_results.json').exists():
        with open('results/standard_vgg_results.json', 'r') as f:
            std_results = json.load(f)
        with open('results/vgg_batchnorm_results.json', 'r') as f:
            bn_results = json.load(f)
        
        summary['Standard Comparison'] = {
            'Standard VGG Test Error': f"{100 - std_results.get('best_test_acc', 0):.2f}%",
            'BatchNorm VGG Test Error': f"{100 - bn_results.get('best_test_acc', 0):.2f}%",
            'Performance Improvement': f"{(100 - bn_results.get('best_test_acc', 0)) - (100 - std_results.get('best_test_acc', 0)):.2f}%"
        }
    
    # Create the summary figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table data
    table_data = []
    table_data.append(['Experiment Type', 'Metric', 'Result'])
    
    for exp_type, results in summary.items():
        for metric, value in results.items():
            table_data.append([exp_type, metric, value])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('BatchNorm Experiment Results Summary', fontsize=16, pad=20)
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    print("Starting to generate all visualizations...")
    
    # Make sure the results directory exists
    Path('results').mkdir(exist_ok=True)
    
    # Generate the charts
    try:
        plot_training_comparison()
    except Exception as e:
        print(f"Failed to generate the training comparison plot: {e}")
    
    try:
        plot_multi_lr_results()
    except Exception as e:
        print(f"Failed to generate the multi-learning-rate comparison plot: {e}")
    
    try:
        plot_gradient_analysis()
    except Exception as e:
        print(f"Failed to generate the gradient analysis plot: {e}")
    
    try:
        plot_filter_visualization()
    except Exception as e:
        print(f"Failed to generate the filter visualization: {e}")
    
    try:
        plot_optimization_results()
    except Exception as e:
        print(f"Failed to generate the optimization strategy comparison plot: {e}")
    
    try:
        generate_summary_table()
    except Exception as e:
        print(f"Failed to generate the summary table: {e}")
    
    print("\nAll visualizations generated!")
    print("Results are saved in the results/ directory")

if __name__ == "__main__":
    main() 