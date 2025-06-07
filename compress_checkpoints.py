#!/usr/bin/env python3
"""
压缩checkpoint文件 - 只保留模型权重，移除优化器状态
这可以减少文件大小30-60%，适用于推理部署
"""

import os
import torch
import glob
from pathlib import Path

def compress_checkpoint(checkpoint_path, output_path=None):
    """
    压缩checkpoint文件，只保留model.state_dict()
    
    Args:
        checkpoint_path: 原始checkpoint文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    print(f"处理文件: {checkpoint_path}")
    
    # 加载原始checkpoint
    try:
        # 修复：添加 weights_only=False 来兼容包含numpy数据的checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"  原始大小: {original_size:.2f} MB")
        
        # 检查checkpoint结构
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 只保留模型权重
                compressed_checkpoint = {
                    'model_state_dict': checkpoint['model_state_dict']
                }
                print(f"  发现完整checkpoint，提取model_state_dict")
            elif 'state_dict' in checkpoint:
                # 只保留模型权重
                compressed_checkpoint = {
                    'state_dict': checkpoint['state_dict']
                }
                print(f"  发现完整checkpoint，提取state_dict")
            else:
                # 假设整个字典就是state_dict
                compressed_checkpoint = checkpoint
                print(f"  假设整个文件就是state_dict")
        else:
            # 如果不是字典，假设就是state_dict
            compressed_checkpoint = checkpoint
            print(f"  文件不是字典格式，保持原样")
        
        # 确定输出路径
        if output_path is None:
            output_path = checkpoint_path
        
        # 保存压缩后的checkpoint
        torch.save(compressed_checkpoint, output_path)
        
        # 计算压缩后大小
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        reduction = (1 - compressed_size / original_size) * 100
        
        print(f"  压缩后大小: {compressed_size:.2f} MB")
        print(f"  减少: {reduction:.1f}%")
        print(f"  ✅ 完成")
        
        return compressed_size, reduction
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None, None

def find_and_compress_checkpoints(project_dir):
    """
    查找并压缩项目目录中的所有checkpoint文件
    """
    project_path = Path(project_dir)
    
    # 查找所有.pt文件
    checkpoint_patterns = [
        "**/*checkpoint*.pt",
        "**/*model*.pt",
        "**/*.pth",
        "**/*.ckpt"
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints = list(project_path.glob(pattern))
        all_checkpoints.extend(checkpoints)
    
    # 去重
    all_checkpoints = list(set(all_checkpoints))
    
    if not all_checkpoints:
        print("❌ 未找到任何checkpoint文件")
        return
    
    print(f"🔍 找到 {len(all_checkpoints)} 个checkpoint文件:")
    for cp in all_checkpoints:
        print(f"  - {cp}")
    
    print("\n🚀 开始压缩...")
    
    total_original_size = 0
    total_compressed_size = 0
    successful_compressions = 0
    
    for checkpoint_path in all_checkpoints:
        compressed_size, reduction = compress_checkpoint(str(checkpoint_path))
        
        if compressed_size is not None:
            original_size = os.path.getsize(str(checkpoint_path)) / (1024 * 1024)
            total_original_size += original_size
            total_compressed_size += compressed_size
            successful_compressions += 1
        
        print()  # 空行分隔
    
    # 总结
    if successful_compressions > 0:
        total_reduction = (1 - total_compressed_size / total_original_size) * 100
        print(f"📊 压缩总结:")
        print(f"  处理文件数: {successful_compressions}/{len(all_checkpoints)}")
        print(f"  原始总大小: {total_original_size:.2f} MB")
        print(f"  压缩后总大小: {total_compressed_size:.2f} MB")
        print(f"  总体减少: {total_reduction:.1f}%")
        print(f"  节省空间: {total_original_size - total_compressed_size:.2f} MB")
    
    print("\n✅ 压缩完成！现在这些文件只包含模型权重，适合推理部署。")

if __name__ == "__main__":
    project_dir = "/Users/tianrunliao/Desktop/pj2 submission"
    
    print("🎯 Checkpoint压缩工具")
    print("📁 项目目录:", project_dir)
    print("🎯 目标: 只保留model.state_dict()，移除优化器状态")
    print("💡 预期减少: 30-60% 文件大小")
    print()
    
    find_and_compress_checkpoints(project_dir)