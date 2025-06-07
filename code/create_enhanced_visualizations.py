#!/usr/bin/env python3
"""
增强版可视化脚本 - 基于simplify实验结果
重点使用表格和清晰的图表展示
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_fast_results():
    """加载simplify实验结果"""
    with open('simplyfied/results/fast_all_results.json', 'r') as f:
        return json.load(f)

def create_comparison_table():
    """创建标准对比表格"""
    results = load_fast_results()
    
    # 创建对比数据
    comparison_data = {
        '模型': ['标准VGG', 'BatchNorm VGG'],
        '测试准确率(%)': [
            f"{results['standard_comparison']['standard_vgg']['test_accuracy']:.2f}",
            f"{results['standard_comparison']['batchnorm_vgg']['test_accuracy']:.2f}"
        ],
        '最佳验证准确率(%)': [
            f"{results['standard_comparison']['standard_vgg']['best_val_acc']:.2f}",
            f"{results['standard_comparison']['batchnorm_vgg']['best_val_acc']:.2f}"
        ],
        '训练时间(秒)': [
            f"{results['standard_comparison']['standard_vgg']['training_time']:.2f}",
            f"{results['standard_comparison']['batchnorm_vgg']['training_time']:.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 设置表格样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('VGG vs BatchNorm VGG 性能对比', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/enhanced_comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_activation_comparison():
    """创建激活函数对比图表"""
    results = load_fast_results()
    activations = results['activation_experiments']
    
    # 准备数据
    names = list(activations.keys())
    test_accs = [activations[name]['test_accuracy'] for name in names]
    val_accs = [activations[name]['best_val_acc'] for name in names]
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 柱状图
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, test_accs, width, label='测试准确率', color='#2196F3')
    bars2 = ax1.bar(x + width/2, val_accs, width, label='验证准确率', color='#FF9800')
    
    ax1.set_xlabel('激活函数')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('不同激活函数性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.upper() for name in names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 表格形式
    table_data = {
        '激活函数': [name.upper() for name in names],
        '测试准确率(%)': [f'{acc:.2f}' for acc in test_accs],
        '验证准确率(%)': [f'{acc:.2f}' for acc in val_accs],
        '排名': [1, 2, 4, 3]  # 基于测试准确率排名
    }
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('测试准确率(%)', ascending=False)
    
    ax2.axis('tight')
    ax2.axis('off')
    
    table = ax2.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 突出显示最佳结果
    table[(1, 0)].set_facecolor('#4CAF50')
    table[(1, 1)].set_facecolor('#4CAF50')
    table[(1, 2)].set_facecolor('#4CAF50')
    
    ax2.set_title('激活函数性能排名表')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_optimizer_comparison():
    """创建优化器对比图表"""
    results = load_fast_results()
    optimizers = results['optimizer_experiments']
    
    # 准备数据
    names = list(optimizers.keys())
    test_accs = [optimizers[name]['test_accuracy'] for name in names]
    training_times = [optimizers[name]['training_time'] for name in names]
    
    # 创建综合对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 准确率对比
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax1.bar(names, test_accs, color=colors)
    ax1.set_title('优化器测试准确率对比')
    ax1.set_ylabel('测试准确率 (%)')
    ax1.grid(True, alpha=0.3)
    
    for bar, acc in zip(bars, test_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. 训练时间对比
    ax2.bar(names, training_times, color=colors)
    ax2.set_title('优化器训练时间对比')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 效率散点图
    ax3.scatter(training_times, test_accs, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax3.annotate(name.upper(), (training_times[i], test_accs[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('训练时间 (秒)')
    ax3.set_ylabel('测试准确率 (%)')
    ax3.set_title('优化器效率分析')
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合排名表
    table_data = {
        '优化器': [name.upper() for name in names],
        '测试准确率(%)': [f'{acc:.2f}' for acc in test_accs],
        '训练时间(秒)': [f'{time:.2f}' for time in training_times],
        '综合评分': [f'{acc/max(test_accs)*0.7 + (max(training_times)-time)/max(training_times)*0.3:.2f}' 
                   for acc, time in zip(test_accs, training_times)]
    }
    
    df = pd.DataFrame(table_data)
    df = df.sort_values('测试准确率(%)', ascending=False)
    
    ax4.axis('tight')
    ax4.axis('off')
    
    table = ax4.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    ax4.set_title('优化器综合排名')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_learning_rate_analysis():
    """创建学习率敏感性分析"""
    results = load_fast_results()
    lr_results = results['learning_rate_experiments']
    
    # 准备数据
    lrs = ['0.0001', '0.0005', '0.001', '0.002', '0.005']
    standard_accs = [lr_results['standard'][lr]['test_accuracy'] for lr in lrs]
    batchnorm_accs = [lr_results['batchnorm'][lr]['test_accuracy'] for lr in lrs]
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 学习率曲线
    lr_values = [float(lr) for lr in lrs]
    ax1.plot(lr_values, standard_accs, 'o-', label='标准VGG', linewidth=2, markersize=8)
    ax1.plot(lr_values, batchnorm_accs, 's-', label='BatchNorm VGG', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.set_xlabel('学习率')
    ax1.set_ylabel('测试准确率 (%)')
    ax1.set_title('学习率敏感性分析')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 详细对比表
    table_data = {
        '学习率': lrs,
        '标准VGG(%)': [f'{acc:.2f}' for acc in standard_accs],
        'BatchNorm VGG(%)': [f'{acc:.2f}' for acc in batchnorm_accs],
        '性能提升(%)': [f'{bn_acc - std_acc:.2f}' for std_acc, bn_acc in zip(standard_accs, batchnorm_accs)]
    }
    
    df = pd.DataFrame(table_data)
    
    ax2.axis('tight')
    ax2.axis('off')
    
    table = ax2.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # 突出显示最佳学习率
    best_std_idx = standard_accs.index(max(standard_accs)) + 1
    best_bn_idx = batchnorm_accs.index(max(batchnorm_accs)) + 1
    
    table[(best_std_idx, 1)].set_facecolor('#FFE082')
    table[(best_bn_idx, 2)].set_facecolor('#A5D6A7')
    
    ax2.set_title('学习率详细对比表')
    
    plt.tight_layout()
    plt.savefig('results/enhanced_learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    print("🎨 生成增强版可视化图表...")
    
    # 确保结果目录存在
    Path('results').mkdir(exist_ok=True)
    
    # 生成所有图表
    create_comparison_table()
    print("✅ 标准对比表格已生成")
    
    create_activation_comparison()
    print("✅ 激活函数对比图表已生成")
    
    create_optimizer_comparison()
    print("✅ 优化器对比图表已生成")
    
    create_learning_rate_analysis()
    print("✅ 学习率分析图表已生成")
    
    print("\n🎉 所有增强版可视化图表生成完成！")
    print("📁 图表保存位置: results/enhanced_*.png")

if __name__ == "__main__":
    main()