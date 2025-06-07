import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import math
from pathlib import Path

# 激活函数部分
def get_activation(name):
    """获取激活函数"""
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'swish': lambda: Swish(),
        'mish': lambda: Mish()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"不支持的激活函数: {name}")
    
    return activations[name.lower()]()

class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """Mish激活函数: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 模型组件
class ConvBlock(nn.Module):
    """基础卷积块：Conv2d + 激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 activation='relu', use_batchnorm=False, dropout_rate=0):
        super(ConvBlock, self).__init__()
        
        layers = []
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, bias=not use_batchnorm))
        
        # 批归一化层（可选）
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # 激活函数
        if activation:
            layers.append(get_activation(activation))
            
        # Dropout层（可选）
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """残差块：两个卷积层 + 残差连接"""
    def __init__(self, channels, activation='relu', use_batchnorm=False, dropout_rate=0):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, padding=1,
                              activation=activation, use_batchnorm=use_batchnorm, dropout_rate=0)
        
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, padding=1,
                              activation=None, use_batchnorm=use_batchnorm, dropout_rate=0)
        
        self.activation = get_activation(activation) if activation else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual  # 残差连接
        
        if self.activation:
            out = self.activation(out)
            
        if self.dropout:
            out = self.dropout(out)
            
        return out

class BottleneckBlock(nn.Module):
    """瓶颈块：3个卷积层（降维、特征提取、升维）"""
    def __init__(self, in_channels, mid_channels, out_channels, 
                 activation='relu', use_batchnorm=False, dropout_rate=0):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0,
                              activation=activation, use_batchnorm=use_batchnorm)
        
        self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1,
                              activation=activation, use_batchnorm=use_batchnorm)
        
        self.conv3 = ConvBlock(mid_channels, out_channels, kernel_size=1, padding=0,
                              activation=None, use_batchnorm=use_batchnorm)
        
        # 如果输入输出通道数不同，需要一个1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0,
                                    activation=None, use_batchnorm=use_batchnorm)
        
        self.activation = get_activation(activation) if activation else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual  # 残差连接
        
        if self.activation:
            out = self.activation(out)
            
        if self.dropout:
            out = self.dropout(out)
            
        return out

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 8 * 8)
        x = self.dropout(self.relu3(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x

class SimpleModel(nn.Module):
    """简单的CNN模型，根据配置动态组合不同模块"""
    def __init__(self, in_channels=3, num_classes=10, block_type='conv', 
                 num_blocks=4, base_channels=64, activation='relu',
                 use_batchnorm=False, dropout_rate=0):
        super(SimpleModel, self).__init__()
        
        self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=3, padding=1,
                                     activation=activation, use_batchnorm=use_batchnorm)
        
        # 动态创建模型主体部分
        blocks = []
        current_channels = base_channels
        
        for i in range(num_blocks):
            if block_type == 'conv':
                blocks.append(ConvBlock(current_channels, current_channels * 2, 
                                        activation=activation,
                                        use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
                current_channels *= 2
            elif block_type == 'residual':
                blocks.append(ResidualBlock(current_channels, 
                                           activation=activation,
                                           use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
                # 每两个残差块后增加通道数
                if i % 2 == 1:
                    blocks.append(ConvBlock(current_channels, current_channels * 2, 
                                          activation=activation,
                                          use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
                    current_channels *= 2
            elif block_type == 'bottleneck':
                blocks.append(BottleneckBlock(current_channels, current_channels // 2, current_channels * 2,
                                             activation=activation,
                                             use_batchnorm=use_batchnorm, dropout_rate=dropout_rate))
                current_channels *= 2
        
        blocks.append(nn.MaxPool2d(2))  # 添加最后的池化层
        self.blocks = nn.Sequential(*blocks)
        
        # 全局平均池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # 计算参数量
        self.param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGA(nn.Module):
    """VGG-A模型（VGG11的别名）"""
    def __init__(self, in_channels=3, num_classes=10, activation='relu', 
                 use_batchnorm=False, dropout_rate=0.5):
        super(VGGA, self).__init__()
        
        # VGG-A配置: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        
        self.features = self._make_layers(cfg, in_channels, activation, use_batchnorm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def _make_layers(self, cfg, in_channels, activation, use_batchnorm):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if use_batchnorm:
                    layers += [conv2d, nn.BatchNorm2d(v), get_activation(activation)]
                else:
                    layers += [conv2d, get_activation(activation)]
                in_channels = v
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 自定义损失函数和优化器部分
class CustomLoss:
    """自定义损失函数集合"""
    
    @staticmethod
    def cross_entropy_with_label_smoothing(outputs, targets, smoothing=0.1):
        """带标签平滑的交叉熵损失"""
        batch_size = targets.size(0)
        num_classes = outputs.size(1)
        
        # 转换为one-hot编码
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # 应用标签平滑
        targets_smooth = (1 - smoothing) * targets_one_hot + smoothing / num_classes
        
        # 计算损失
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss
    
    @staticmethod
    def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
        """Focal Loss，用于处理类别不平衡问题"""
        num_classes = outputs.size(1)
        
        # 转换为one-hot编码
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # 计算softmax概率
        probs = F.softmax(outputs, dim=1)
        
        # 计算Focal Loss
        pt = (targets_one_hot * probs).sum(1)  # 目标类别的概率
        focal_weight = (1 - pt) ** gamma
        
        # 使用alpha平衡正负样本
        if alpha is not None:
            alpha_weight = targets_one_hot * alpha + (1 - targets_one_hot) * (1 - alpha)
            focal_weight = focal_weight * alpha_weight.sum(dim=1)
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # 应用权重
        loss = (focal_weight * ce_loss).mean()
        
        return loss

def get_optimizer(name, parameters, **kwargs):
    """获取优化器
    
    Args:
        name: 优化器名称
        parameters: 模型参数
        **kwargs: 优化器的附加参数
    
    Returns:
        优化器实例
    """
    optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"不支持的优化器: {name}")
    
    return optimizers[name.lower()](parameters, **kwargs)

def get_loss_function(name, **kwargs):
    """获取损失函数
    
    Args:
        name: 损失函数名称
        **kwargs: 损失函数的附加参数
    
    Returns:
        损失函数实例
    """
    loss_functions = {
        'cross_entropy': nn.CrossEntropyLoss,
        'label_smoothing': lambda: lambda outputs, targets: CustomLoss.cross_entropy_with_label_smoothing(
            outputs, targets, smoothing=kwargs.get('smoothing', 0.1)
        ),
        'focal': lambda: lambda outputs, targets: CustomLoss.focal_loss(
            outputs, targets, gamma=kwargs.get('gamma', 2.0), alpha=kwargs.get('alpha', 0.25)
        ),
        'mse': nn.MSELoss,
        'l1': nn.L1Loss
    }
    
    if name.lower() not in loss_functions:
        raise ValueError(f"不支持的损失函数: {name}")
    
    return loss_functions[name.lower()]()

# 数据加载相关函数
def prepare_data(batch_size=64, test_split=0.2, dataset='cifar10'):
    """准备数据集和数据加载器"""
    # 数据转换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # 加载数据集
    if dataset.lower() == 'cifar10':
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=val_transform
        )
        num_classes = 10
    elif dataset.lower() == 'cifar100':
        full_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=val_transform
        )
        num_classes = 100
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器（Windows兼容：num_workers=0）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, num_classes

# 训练优化器类
class TrainingOptimizer:
    def __init__(self, model, optimizer, train_loader, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.grad_accum_steps = 1
        self.scaler = None  # 用于混合精度训练
        
    def enable_mixed_precision(self):
        """启用混合精度训练"""
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("已启用混合精度训练")
        else:
            print("CUDA不可用，无法启用混合精度训练")
    
    def optimize_memory_format(self):
        """优化内存格式以提高性能"""
        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
            print("已优化内存格式")
    
    def set_gradient_accumulation(self, steps=2):
        """设置梯度累积步数"""
        self.grad_accum_steps = steps
        print(f"已设置梯度累积，每{steps}步更新一次")
    
    def optimize_dataloader(self, num_workers=4, pin_memory=True):
        """优化数据加载器"""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print(f"已优化数据加载器: num_workers={num_workers}, pin_memory={pin_memory}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 如果使用通道优先格式
            if hasattr(torch.cuda, 'amp') and self.device.type == 'cuda':
                inputs = inputs.to(memory_format=torch.channels_last)
            
            # 是否使用混合精度
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.grad_accum_steps  # 梯度累积时缩小损失
                
                self.scaler.scale(loss).backward()
                
                # 梯度累积
                if (i + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps  # 梯度累积时缩小损失
                
                loss.backward()
                
                # 梯度累积
                if (i + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.grad_accum_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

# 训练器类
class Trainer:
    def __init__(self, model, config, device=None):
        """训练器类，负责模型的训练、验证和日志记录"""
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 从配置中读取参数
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.batch_size = config.get('batch_size', 128)
        self.dataset_name = config.get('dataset', 'cifar10')
        self.optimizer_name = config.get('optimizer', 'adam')
        self.criterion_name = config.get('criterion', 'cross_entropy')
        self.experiment_name = config.get('experiment_name', 'default_experiment')
        self.use_optimization = config.get('use_optimization', False)
        self.resume_training = config.get('resume_training', False)
        
        # 准备数据
        self.train_loader, self.val_loader, self.test_loader, num_classes = prepare_data(
            batch_size=self.batch_size, 
            dataset=self.dataset_name
        )
        
        # 设置优化器
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        # 获取优化器
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), **optimizer_params)
        
        # 获取损失函数
        self.criterion = get_loss_function(self.criterion_name)
        
        # 训练优化器
        if self.use_optimization:
            self.training_optimizer = TrainingOptimizer(
                self.model, self.optimizer, self.train_loader, self.criterion, self.device
            )
            self.training_optimizer.enable_mixed_precision()
            self.training_optimizer.optimize_memory_format()
            self.training_optimizer.set_gradient_accumulation(steps=2)
        
        # 存储训练记录
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.start_epoch = 0
        
        # 尝试加载检查点
        if self.resume_training:
            self._load_checkpoint()
    
    def train_epoch(self):
        """训练一个完整的epoch"""
        if self.use_optimization:
            epoch_loss, epoch_acc = self.training_optimizer.train_epoch(self.current_epoch + 1)
        else:
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Train]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """验证一个完整的epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Val]"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        # 保存最佳模型
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self._save_model('best_model.pt')
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self):
        """保存训练检查点，用于断点续训"""
        checkpoint = {
            'epoch': self.current_epoch + 1,  # 保存下一个要训练的epoch
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        save_dir = os.path.join('results', self.experiment_name, 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存：{checkpoint_path}，当前epoch：{self.current_epoch}")
    
    def _load_checkpoint(self):
        """加载训练检查点，用于断点续训"""
        checkpoint_path = os.path.join('results', self.experiment_name, 'checkpoints', 'checkpoint.pt')
        
        if os.path.exists(checkpoint_path):
            print(f"加载检查点：{checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练历史
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            self.best_val_acc = checkpoint['best_val_acc']
            
            # 设置起始epoch
            self.start_epoch = checkpoint['epoch']
            print(f"从epoch {self.start_epoch}继续训练")
        else:
            print("未找到检查点，将从头开始训练")
    
    def train(self):
        """完整的训练过程"""
        print(f"开始训练 | 模型: {type(self.model).__name__} | 设备: {self.device}")
        print(f"配置: 数据集={self.dataset_name}, 优化器={self.optimizer_name}, 学习率={self.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_loss, train_acc = self.train_epoch()
            
            # 验证阶段
            val_loss, val_acc = self.validate_epoch()
            
            # 打印统计信息
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存检查点
            self._save_checkpoint()
            
            # 绘制当前训练曲线
            self._plot_training_curves()
        
        # 保存最终模型
        self._save_model('final_model.pt')
        
        # 保存训练历史
        self._save_history()
        
        # 训练完成
        train_time = time.time() - start_time
        print(f"训练完成! 总时间: {train_time:.2f}秒")
        print(f"最佳验证集准确率: {self.best_val_acc:.2f}%")
    
    def _save_model(self, filename):
        """保存模型"""
        save_dir = os.path.join('results', self.experiment_name, 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))
    
    def _save_history(self):
        """保存训练历史"""
        history = {
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        save_dir = os.path.join('results', self.experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        save_dir = os.path.join('results', self.experiment_name, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        
        # 准确率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, 'b-', label='训练准确率')
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, 'r-', label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
        
        # 损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='训练损失')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='验证损失')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        
        plt.close('all')
    
    def test_model(self):
        """在测试集上测试模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="测试"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        print(f"测试集准确率: {test_acc:.2f}%")
        return test_acc

# 配置加载器函数
def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 

# 添加VGG_A_BatchNorm模型
class VGG_A_BatchNorm(nn.Module):
    """
    带有BatchNorm的VGG_A模型
    
    实现了和VGG_A相同的网络结构，但在每个卷积层后添加了BatchNorm2d层
    输入假设为32x32x3的CIFAR10图像
    """
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        
        self.features = nn.Sequential(
            # stage 1
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # stage 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        if init_weights:
            self._init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias) 

def get_accuracy(model, data_loader, device):
    """
    计算模型在给定数据加载器上的准确率
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备 (CPU或CUDA)
        
    Returns:
        float: 准确率百分比
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total 