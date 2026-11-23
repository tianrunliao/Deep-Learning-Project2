#!/usr/bin/env python3
"""
Comprehensive experiment script for the effect of BatchNorm on VGG models
Author: Tianrun Liao
Student ID: 22300680285

This script combines all experiment functionality, including:
1. Standard comparison experiment (completed, uses the existing checkpoints)
2. Multi-learning-rate experiment (completed, uses the existing results)
3. Optimization strategy experiments (new experiments that need to be run)
4. Visualization generation
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import the core functionality module
from functions import (
    VGGA, VGG_A_BatchNorm, prepare_data, 
    get_activation, get_optimizer, get_loss_function,
    Trainer, load_config
)

def check_existing_results():
    """Check the existing experiment results"""
    existing = {
        'standard_comparison': False,
        'multi_lr': False,
        'activation': False,
        'loss_function': False,
        'optimizer': False,
        'visualization': False
    }
    
    # Check the standard comparison experiment
    if (Path('results/standard_vgg_results.json').exists() and 
        Path('results/vgg_batchnorm_results.json').exists()):
        existing['standard_comparison'] = True
        print("✓ Standard comparison experiment completed")
    
    # Check the multi-learning-rate experiment
    if Path('results/multi_lr_experiment').exists():
        existing['multi_lr'] = True
        print("✓ Multi-learning-rate experiment completed")
    
    # Check the optimization strategy experiments
    opt_path = Path('results/optimization_experiments')
    if opt_path.exists():
        if (opt_path / 'activation_results.json').exists():
            existing['activation'] = True
            print("✓ Activation function experiment completed")
        if (opt_path / 'loss_function_results.json').exists():
            existing['loss_function'] = True
            print("✓ Loss function experiment completed")
        if (opt_path / 'optimizer_results.json').exists():
            existing['optimizer'] = True
            print("✓ Optimizer experiment completed")
    
    # Check the visualizations
    if (Path('results/loss_comparison.png').exists() and
        Path('results/gradient_analysis.png').exists()):
        existing['visualization'] = True
        print("✓ Visualizations generated")
    
    return existing

def load_trained_model(model_type='standard', checkpoint_path=None):
    """Load a trained model"""
    if model_type == 'standard':
        model = VGGA(use_batchnorm=False)
        default_path = 'results/checkpoints/standard_vgg_checkpoint.pt'
    else:
        model = VGG_A_BatchNorm()
        default_path = 'results/checkpoints/vgg_batchnorm_checkpoint.pt'
    
    checkpoint_path = checkpoint_path or default_path
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model: {checkpoint_path}")
        return model, True
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return model, False

def run_missing_experiments(existing):
    """Run the missing experiments"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Prepare the data
    train_loader, val_loader, test_loader, _ = prepare_data(batch_size=128)
    
    # Run the activation function experiments
    if not existing['activation']:
        print("\nStarting the activation function comparison experiments...")
        run_activation_experiments(train_loader, val_loader, test_loader, device)
    
    # Run the loss function experiments
    if not existing['loss_function']:
        print("\nStarting the loss function comparison experiments...")
        run_loss_function_experiments(train_loader, val_loader, test_loader, device)
    
    # Run the optimizer experiments
    if not existing['optimizer']:
        print("\nStarting the optimizer comparison experiments...")
        run_optimizer_experiments(train_loader, val_loader, test_loader, device)

def run_activation_experiments(train_loader, val_loader, test_loader, device):
    """Run the activation function comparison experiments"""
    from optimization_experiments import train_model
    
    activations = ['relu', 'leaky_relu', 'elu', 'swish']
    results = {}
    
    for use_bn in [False, True]:
        model_type = "BatchNorm VGG" if use_bn else "Standard VGG"
        results[model_type] = {}
        
        for activation in activations:
            print(f"\nTraining {model_type} with {activation}")
            
            # For the BatchNorm model we use a fixed ReLU, because VGG_A_BatchNorm does not support custom activation functions
            if use_bn:
                if activation != 'relu':
                    print(f"Skipping {activation} (the BatchNorm model only supports ReLU)")
                    results[model_type][activation] = {
                        'best_test_error': 'N/A',
                        'best_epoch': 'N/A'
                    }
                    continue
                model = VGG_A_BatchNorm()
            else:
                model = VGGA(activation=activation, use_batchnorm=False)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            result = train_model(
                model, train_loader, val_loader, test_loader,
                optimizer, criterion, device, epochs=20,
                model_name=f"{model_type.replace(' ', '_')}_{activation}"
            )
            
            results[model_type][activation] = result
            print(f"Done: test error {result['best_test_error']:.2f}%")
    
    # Save the results
    os.makedirs('results/optimization_experiments', exist_ok=True)
    with open('results/optimization_experiments/activation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def run_loss_function_experiments(train_loader, val_loader, test_loader, device):
    """Run the loss function comparison experiments"""
    from optimization_experiments import train_model, train_model_with_l1
    
    loss_configs = [
        ('cross_entropy', nn.CrossEntropyLoss(), 0, 1e-4),
        ('cross_entropy_l1', nn.CrossEntropyLoss(), 1e-4, 0),
        ('cross_entropy_l2', nn.CrossEntropyLoss(), 0, 1e-3),
        ('label_smoothing', get_loss_function('label_smoothing', smoothing=0.1), 0, 1e-4)
    ]
    
    results = {}
    
    for use_bn in [False, True]:
        model_type = "BatchNorm VGG" if use_bn else "Standard VGG"
        results[model_type] = {}
        
        for loss_name, criterion, l1_lambda, weight_decay in loss_configs:
            print(f"\nTraining {model_type} with {loss_name}")
            
            if use_bn:
                model = VGG_A_BatchNorm()
            else:
                model = VGGA(use_batchnorm=False)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            
            if l1_lambda > 0:
                result = train_model_with_l1(
                    model, train_loader, val_loader, test_loader,
                    optimizer, criterion, device, l1_lambda,
                    epochs=20, model_name=f"{model_type.replace(' ', '_')}_{loss_name}"
                )
            else:
                result = train_model(
                    model, train_loader, val_loader, test_loader,
                    optimizer, criterion, device, epochs=20,
                    model_name=f"{model_type.replace(' ', '_')}_{loss_name}"
                )
            
            results[model_type][loss_name] = result
            print(f"Done: test error {result['best_test_error']:.2f}%")
    
    # Save the results
    with open('results/optimization_experiments/loss_function_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def run_optimizer_experiments(train_loader, val_loader, test_loader, device):
    """Run the optimizer comparison experiments"""
    from optimization_experiments import train_model
    
    optimizer_configs = [
        ('SGD', lambda p: optim.SGD(p, lr=0.01, weight_decay=1e-4)),
        ('SGD+Momentum', lambda p: optim.SGD(p, lr=0.01, momentum=0.9, weight_decay=1e-4)),
        ('Adam', lambda p: optim.Adam(p, lr=1e-3, weight_decay=1e-4)),
        ('AdamW', lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-4)),
        ('RMSprop', lambda p: optim.RMSprop(p, lr=1e-3, weight_decay=1e-4))
    ]
    
    results = {}
    
    for use_bn in [False, True]:
        model_type = "BatchNorm VGG" if use_bn else "Standard VGG"
        results[model_type] = {}
        
        for opt_name, opt_func in optimizer_configs:
            print(f"\nTraining {model_type} with {opt_name}")
            
            if use_bn:
                model = VGG_A_BatchNorm()
            else:
                model = VGGA(use_batchnorm=False)
            
            optimizer = opt_func(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            epochs = 30 if 'SGD' in opt_name else 20
            
            result = train_model(
                model, train_loader, val_loader, test_loader,
                optimizer, criterion, device, epochs=epochs,
                model_name=f"{model_type.replace(' ', '_')}_{opt_name}"
            )
            
            results[model_type][opt_name] = result
            print(f"Done: test error {result['best_test_error']:.2f}%")
    
    # Save the results
    with open('results/optimization_experiments/optimizer_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def generate_all_visualizations():
    """Generate all visualization charts"""
    print("\nGenerating visualization charts...")
    
    # Generate the charts
    os.system('python3 visualization.py')
    
    print("✓ All visualizations completed")

def generate_final_report():
    """Generate the final experiment report"""
    print("\nGenerating the final report...")
    
    report_content = f"""BatchNorm对VGG模型影响的实验研究
作者：廖天润
学号：22300680285
代码仓库：https://github.com/tianrunliao/Deep-Learning-Project2

一、项目概述
本项目深入研究了批量归一化（Batch Normalization）技术对VGG网络在CIFAR-10数据集上的影响。

二、实验结果
存档的实验日志：results/multi_lr_experiment/standard_vgg/results.json 与 results/multi_lr_experiment/vgg_batchnorm/results.json。

三、项目文件说明

1. 核心代码:
   - functions.py: 模型定义和训练功能
   - run_all_experiments.py: 主实验脚本
   - visualization.py: 可视化生成

2. 实验结果:
   - results/: 所有实验结果和图表
   - results/checkpoints/: 训练好的模型
   - results/optimization_experiments/: 优化策略实验结果

3. 配置文件:
   - configs/: 模型配置文件

四、复现说明

1. 环境要求:
   - Python 3.7+
   - PyTorch 1.8+
   - CUDA (可选)

2. 运行实验:
   python3 run_all_experiments.py

3. 生成可视化:
   python3 visualization.py

五、参考文献
[1] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift.
[2] Santurkar, S., et al. (2018). How does batch normalization help optimization?
[3] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.
"""
    
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("✓ Report generated: experiment_report.txt")

def main():
    """Main function"""
    print("="*60)
    print("Comprehensive experiments on the effect of BatchNorm on VGG models")
    print("Author: Tianrun Liao (22300680285)")
    print("="*60)
    
    # Check the existing results
    print("\nChecking existing experiment results...")
    existing = check_existing_results()
    
    # Run the missing experiments
    missing_count = sum(1 for v in existing.values() if not v)
    if missing_count > 0:
        print(f"\nNeed to run {missing_count} missing experiments")
        run_missing_experiments(existing)
    else:
        print("\nAll experiments are already completed!")
    
    # Generate the visualizations
    if not existing['visualization']:
        generate_all_visualizations()
    
    # Generate the final report
    generate_final_report()
    
    print("\nAll experiments finished!")
    print("See experiment_report.txt for the full report")

if __name__ == "__main__":
    main() 