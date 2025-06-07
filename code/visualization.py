#!/usr/bin/env python3
"""
综合可视化脚本
整合所有图表生成功能，使用中文标签
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import torch
from functions import VGGA, VGG_A_BatchNorm

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_training_comparison():
    """绘制标准对比实验的训练曲线"""
    print("生成训练对比图...")
    
    # 加载数据
    with open('results/standard_vgg_results.json', 'r') as f:
        std_results = json.load(f)
    with open('results/vgg_batchnorm_results.json', 'r') as f:
        bn_results = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 损失曲线
    ax1.plot(std_results['train_losses'], label='标准VGG', linewidth=2)
    ax1.plot(bn_results['train_losses'], label='BatchNorm VGG', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('训练损失', fontsize=12)
    ax1.set_title('训练损失对比', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证准确率曲线
    ax2.plot(std_results['val_accs'], label='标准VGG', linewidth=2)
    ax2.plot(bn_results['val_accs'], label='BatchNorm VGG', linewidth=2)
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('验证准确率 (%)', fontsize=12)
    ax2.set_title('验证准确率对比', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_multi_lr_results():
    """绘制多学习率实验结果"""
    print("生成多学习率对比图...")
    
    lr_dir = Path('results/multi_lr_experiment')
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        
        # 加载标准VGG结果
        std_file = lr_dir / f'standard_vgg_lr_{lr}_results.json'
        if std_file.exists():
            with open(std_file, 'r') as f:
                std_data = json.load(f)
                ax.plot(std_data['train_losses'], label='标准VGG', linewidth=2)
        
        # 加载BatchNorm VGG结果
        bn_file = lr_dir / f'vgg_batchnorm_lr_{lr}_results.json'
        if bn_file.exists():
            with open(bn_file, 'r') as f:
                bn_data = json.load(f)
                ax.plot(bn_data['train_losses'], label='BatchNorm VGG', linewidth=2)
        
        ax.set_title(f'学习率 = {lr}', fontsize=12)
        ax.set_xlabel('训练轮次', fontsize=10)
        ax.set_ylabel('训练损失', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 3)
    
    plt.suptitle('不同学习率下的训练损失对比', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/multi_lr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_gradient_analysis():
    """绘制梯度分析图"""
    print("生成梯度分析图...")
    
    lr_dir = Path('results/multi_lr_experiment')
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 梯度预测性
    for model_type in ['standard_vgg', 'vgg_batchnorm']:
        predictiveness = []
        for lr in learning_rates:
            result_file = lr_dir / f'{model_type}_lr_{lr}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'gradient_predictiveness' in data:
                        # 取平均值
                        avg_pred = np.mean([p for p in data['gradient_predictiveness'] if p is not None])
                        predictiveness.append(avg_pred)
                    else:
                        predictiveness.append(0)
        
        label = 'BatchNorm VGG' if 'batchnorm' in model_type else '标准VGG'
        ax1.plot(range(len(learning_rates)), predictiveness, 'o-', label=label, linewidth=2, markersize=8)
    
    ax1.set_xticks(range(len(learning_rates)))
    ax1.set_xticklabels([f'{lr}' for lr in learning_rates])
    ax1.set_xlabel('学习率', fontsize=12)
    ax1.set_ylabel('梯度预测性 (余弦相似度)', fontsize=12)
    ax1.set_title('梯度预测性对比', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 梯度方差
    for model_type in ['standard_vgg', 'vgg_batchnorm']:
        variances = []
        for lr in learning_rates:
            result_file = lr_dir / f'{model_type}_lr_{lr}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'gradient_variance' in data:
                        # 取对数平均值
                        avg_var = np.mean([np.log10(v) for v in data['gradient_variance'] if v > 0])
                        variances.append(avg_var)
                    else:
                        variances.append(-8)
        
        label = 'BatchNorm VGG' if 'batchnorm' in model_type else '标准VGG'
        ax2.plot(range(len(learning_rates)), variances, 'o-', label=label, linewidth=2, markersize=8)
    
    ax2.set_xticks(range(len(learning_rates)))
    ax2.set_xticklabels([f'{lr}' for lr in learning_rates])
    ax2.set_xlabel('学习率', fontsize=12)
    ax2.set_ylabel('梯度方差 (log10)', fontsize=12)
    ax2.set_title('梯度方差对比', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/gradient_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_filter_visualization():
    """可视化第一层卷积滤波器"""
    print("生成滤波器可视化...")
    
    device = torch.device('cpu')
    
    # 加载模型
    std_model = VGGA(use_batchnorm=False)
    bn_model = VGG_A_BatchNorm()
    
    # 尝试加载训练好的权重
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
    
    # 获取第一层卷积权重
    std_filters = std_model.features[0].weight.data.cpu().numpy()
    bn_filters = bn_model.features[0].weight.data.cpu().numpy()
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 标准VGG滤波器
    n_filters = min(64, std_filters.shape[0])
    grid_size = int(np.sqrt(n_filters))
    
    for i in range(n_filters):
        ax1.subplot(grid_size, grid_size, i+1)
        filter_img = std_filters[i].transpose(1, 2, 0)
        # 归一化到[0, 1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        ax1.imshow(filter_img)
        ax1.axis('off')
    
    ax1.set_title('标准VGG第一层滤波器', fontsize=14)
    
    # BatchNorm VGG滤波器
    for i in range(n_filters):
        ax2.subplot(grid_size, grid_size, i+1)
        filter_img = bn_filters[i].transpose(1, 2, 0)
        # 归一化到[0, 1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
        ax2.imshow(filter_img)
        ax2.axis('off')
    
    ax2.set_title('BatchNorm VGG第一层滤波器', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/filter_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_optimization_results():
    """绘制优化策略实验结果"""
    print("生成优化策略对比图...")
    
    opt_dir = Path('results/optimization_experiments')
    
    # 检查是否有结果文件
    if not opt_dir.exists():
        print("优化策略实验结果不存在，跳过...")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 激活函数对比
    if (opt_dir / 'activation_results.json').exists():
        with open(opt_dir / 'activation_results.json', 'r') as f:
            act_results = json.load(f)
        
        activations = ['relu', 'leaky_relu', 'elu', 'swish']
        std_errors = []
        bn_errors = []
        
        for act in activations:
            if act in act_results.get('标准VGG', {}):
                error = act_results['标准VGG'][act].get('best_test_error', 100)
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
        
        ax1.bar(x - width/2, std_errors, width, label='标准VGG')
        ax1.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax1.set_xlabel('激活函数')
        ax1.set_ylabel('测试误差 (%)')
        ax1.set_title('不同激活函数的测试误差')
        ax1.set_xticks(x)
        ax1.set_xticklabels(activations)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 损失函数对比
    if (opt_dir / 'loss_function_results.json').exists():
        with open(opt_dir / 'loss_function_results.json', 'r') as f:
            loss_results = json.load(f)
        
        loss_names = ['cross_entropy', 'cross_entropy_l1', 'cross_entropy_l2', 'label_smoothing']
        loss_labels = ['CE', 'CE+L1', 'CE+L2', 'LS']
        std_errors = []
        bn_errors = []
        
        for loss in loss_names:
            if loss in loss_results.get('标准VGG', {}):
                std_errors.append(loss_results['标准VGG'][loss].get('best_test_error', 100))
            else:
                std_errors.append(100)
                
            if loss in loss_results.get('BatchNorm VGG', {}):
                bn_errors.append(loss_results['BatchNorm VGG'][loss].get('best_test_error', 100))
            else:
                bn_errors.append(100)
        
        x = np.arange(len(loss_names))
        
        ax2.bar(x - width/2, std_errors, width, label='标准VGG')
        ax2.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax2.set_xlabel('损失函数')
        ax2.set_ylabel('测试误差 (%)')
        ax2.set_title('不同损失函数的测试误差')
        ax2.set_xticks(x)
        ax2.set_xticklabels(loss_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 优化器对比
    if (opt_dir / 'optimizer_results.json').exists():
        with open(opt_dir / 'optimizer_results.json', 'r') as f:
            opt_results = json.load(f)
        
        optimizers = ['SGD', 'SGD+Momentum', 'Adam', 'AdamW', 'RMSprop']
        std_errors = []
        bn_errors = []
        
        for opt in optimizers:
            if opt in opt_results.get('标准VGG', {}):
                std_errors.append(opt_results['标准VGG'][opt].get('best_test_error', 100))
            else:
                std_errors.append(100)
                
            if opt in opt_results.get('BatchNorm VGG', {}):
                bn_errors.append(opt_results['BatchNorm VGG'][opt].get('best_test_error', 100))
            else:
                bn_errors.append(100)
        
        x = np.arange(len(optimizers))
        
        ax3.bar(x - width/2, std_errors, width, label='标准VGG')
        ax3.bar(x + width/2, bn_errors, width, label='BatchNorm VGG')
        ax3.set_xlabel('优化器')
        ax3.set_ylabel('测试误差 (%)')
        ax3.set_title('不同优化器的测试误差')
        ax3.set_xticks(x)
        ax3.set_xticklabels(optimizers, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table():
    """生成实验结果汇总表"""
    print("生成结果汇总表...")
    
    # 收集所有结果
    summary = {
        '标准对比实验': {},
        '多学习率实验': {},
        '优化策略实验': {}
    }
    
    # 标准对比结果
    if Path('results/standard_vgg_results.json').exists():
        with open('results/standard_vgg_results.json', 'r') as f:
            std_results = json.load(f)
        with open('results/vgg_batchnorm_results.json', 'r') as f:
            bn_results = json.load(f)
        
        summary['标准对比实验'] = {
            '标准VGG测试误差': f"{100 - std_results.get('best_test_acc', 0):.2f}%",
            'BatchNorm VGG测试误差': f"{100 - bn_results.get('best_test_acc', 0):.2f}%",
            '性能提升': f"{(100 - bn_results.get('best_test_acc', 0)) - (100 - std_results.get('best_test_acc', 0)):.2f}%"
        }
    
    # 创建汇总图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    table_data.append(['实验类型', '指标', '结果'])
    
    for exp_type, results in summary.items():
        for metric, value in results.items():
            table_data.append([exp_type, metric, value])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置标题行样式
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('BatchNorm实验结果汇总', fontsize=16, pad=20)
    plt.savefig('results/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("开始生成所有可视化...")
    
    # 确保结果目录存在
    Path('results').mkdir(exist_ok=True)
    
    # 生成各种图表
    try:
        plot_training_comparison()
    except Exception as e:
        print(f"训练对比图生成失败: {e}")
    
    try:
        plot_multi_lr_results()
    except Exception as e:
        print(f"多学习率对比图生成失败: {e}")
    
    try:
        plot_gradient_analysis()
    except Exception as e:
        print(f"梯度分析图生成失败: {e}")
    
    try:
        plot_filter_visualization()
    except Exception as e:
        print(f"滤波器可视化生成失败: {e}")
    
    try:
        plot_optimization_results()
    except Exception as e:
        print(f"优化策略对比图生成失败: {e}")
    
    try:
        generate_summary_table()
    except Exception as e:
        print(f"汇总表生成失败: {e}")
    
    print("\n所有可视化生成完成！")
    print("结果保存在 results/ 目录下")

if __name__ == "__main__":
    main() 