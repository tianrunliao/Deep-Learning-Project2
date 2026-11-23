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

# Activation functions
def get_activation(name):
    """Get an activation function"""
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
        raise ValueError(f"Unsupported activation function: {name}")
    
    return activations[name.lower()]()

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Model components
class ConvBlock(nn.Module):
    """Basic convolution block: Conv2d + activation function"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 activation='relu', use_batchnorm=False, dropout_rate=0):
        super(ConvBlock, self).__init__()
        
        layers = []
        # Convolution layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, bias=not use_batchnorm))
        
        # Batch normalization layer (optional)
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # Activation function
        if activation:
            layers.append(get_activation(activation))
            
        # Dropout layer (optional)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual block: two convolution layers + residual connection"""
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
        out += residual  # Residual connection
        
        if self.activation:
            out = self.activation(out)
            
        if self.dropout:
            out = self.dropout(out)
            
        return out

class BottleneckBlock(nn.Module):
    """Bottleneck block: 3 convolution layers (dimension reduction, feature extraction, dimension expansion)"""
    def __init__(self, in_channels, mid_channels, out_channels, 
                 activation='relu', use_batchnorm=False, dropout_rate=0):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0,
                              activation=activation, use_batchnorm=use_batchnorm)
        
        self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=3, padding=1,
                              activation=activation, use_batchnorm=use_batchnorm)
        
        self.conv3 = ConvBlock(mid_channels, out_channels, kernel_size=1, padding=0,
                              activation=None, use_batchnorm=use_batchnorm)
        
        # If the input and output channel counts differ, a 1x1 convolution is needed to match them
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
        
        out += residual  # Residual connection
        
        if self.activation:
            out = self.activation(out)
            
        if self.dropout:
            out = self.dropout(out)
            
        return out

# Model definitions
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
    """A simple CNN model that dynamically combines different blocks based on the configuration"""
    def __init__(self, in_channels=3, num_classes=10, block_type='conv', 
                 num_blocks=4, base_channels=64, activation='relu',
                 use_batchnorm=False, dropout_rate=0):
        super(SimpleModel, self).__init__()
        
        self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=3, padding=1,
                                     activation=activation, use_batchnorm=use_batchnorm)
        
        # Dynamically build the main body of the model
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
                # Increase the channel count after every two residual blocks
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
        
        blocks.append(nn.MaxPool2d(2))  # Add the final pooling layer
        self.blocks = nn.Sequential(*blocks)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # Count the parameters
        self.param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGA(nn.Module):
    """VGG-A model (an alias for VGG11)"""
    def __init__(self, in_channels=3, num_classes=10, activation='relu', 
                 use_batchnorm=False, dropout_rate=0.5):
        super(VGGA, self).__init__()
        
        # VGG-A configuration: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
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

# Custom loss functions and optimizers
class CustomLoss:
    """Collection of custom loss functions"""
    
    @staticmethod
    def cross_entropy_with_label_smoothing(outputs, targets, smoothing=0.1):
        """Cross-entropy loss with label smoothing"""
        batch_size = targets.size(0)
        num_classes = outputs.size(1)
        
        # Convert to one-hot encoding
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (1 - smoothing) * targets_one_hot + smoothing / num_classes
        
        # Compute the loss
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss
    
    @staticmethod
    def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
        """Focal Loss, used to handle class imbalance"""
        num_classes = outputs.size(1)
        
        # Convert to one-hot encoding
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Compute softmax probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Compute the Focal Loss
        pt = (targets_one_hot * probs).sum(1)  # Probability of the target class
        focal_weight = (1 - pt) ** gamma
        
        # Use alpha to balance positive and negative samples
        if alpha is not None:
            alpha_weight = targets_one_hot * alpha + (1 - targets_one_hot) * (1 - alpha)
            focal_weight = focal_weight * alpha_weight.sum(dim=1)
        
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Apply the weights
        loss = (focal_weight * ce_loss).mean()
        
        return loss

def get_optimizer(name, parameters, **kwargs):
    """Get an optimizer
    
    Args:
        name: Optimizer name
        parameters: Model parameters
        **kwargs: Additional arguments for the optimizer
    
    Returns:
        Optimizer instance
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
        raise ValueError(f"Unsupported optimizer: {name}")
    
    return optimizers[name.lower()](parameters, **kwargs)

def get_loss_function(name, **kwargs):
    """Get a loss function
    
    Args:
        name: Loss function name
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function instance
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
        raise ValueError(f"Unsupported loss function: {name}")
    
    return loss_functions[name.lower()]()

# Data loading functions
def prepare_data(batch_size=64, test_split=0.2, dataset='cifar10'):
    """Prepare the datasets and data loaders"""
    # Data transforms
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
    
    # Load the dataset
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
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Split into training and validation sets
    val_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders (Windows-compatible: num_workers=0)
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

# Training optimizer class
class TrainingOptimizer:
    def __init__(self, model, optimizer, train_loader, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.grad_accum_steps = 1
        self.scaler = None  # Used for mixed precision training
        
    def enable_mixed_precision(self):
        """Enable mixed precision training"""
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        else:
            print("CUDA is not available, cannot enable mixed precision training")
    
    def optimize_memory_format(self):
        """Optimize the memory format to improve performance"""
        if torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
            print("Memory format optimized")
    
    def set_gradient_accumulation(self, steps=2):
        """Set the number of gradient accumulation steps"""
        self.grad_accum_steps = steps
        print(f"Gradient accumulation set: update once every {steps} steps")
    
    def optimize_dataloader(self, num_workers=4, pin_memory=True):
        """Optimize the data loader"""
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print(f"Optimized data loader: num_workers={num_workers}, pin_memory={pin_memory}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # If using the channel-priority memory format
            if hasattr(torch.cuda, 'amp') and self.device.type == 'cuda':
                inputs = inputs.to(memory_format=torch.channels_last)
            
            # Whether to use mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.grad_accum_steps  # Scale down the loss for gradient accumulation
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.grad_accum_steps  # Scale down the loss for gradient accumulation
                
                loss.backward()
                
                # Gradient accumulation
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

# Trainer class
class Trainer:
    def __init__(self, model, config, device=None):
        """Trainer class, responsible for model training, validation and logging"""
        self.model = model
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Read parameters from the config
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
        
        # Prepare the data
        self.train_loader, self.val_loader, self.test_loader, num_classes = prepare_data(
            batch_size=self.batch_size, 
            dataset=self.dataset_name
        )
        
        # Set up the optimizer
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        # Get the optimizer
        self.optimizer = get_optimizer(self.optimizer_name, self.model.parameters(), **optimizer_params)
        
        # Get the loss function
        self.criterion = get_loss_function(self.criterion_name)
        
        # Training optimizer
        if self.use_optimization:
            self.training_optimizer = TrainingOptimizer(
                self.model, self.optimizer, self.train_loader, self.criterion, self.device
            )
            self.training_optimizer.enable_mixed_precision()
            self.training_optimizer.optimize_memory_format()
            self.training_optimizer.set_gradient_accumulation(steps=2)
        
        # Store the training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.start_epoch = 0
        
        # Try to load a checkpoint
        if self.resume_training:
            self._load_checkpoint()
    
    def train_epoch(self):
        """Train for one full epoch"""
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
        """Validate for one full epoch"""
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
        
        # Save the best model
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            self._save_model('best_model.pt')
        
        return epoch_loss, epoch_acc
    
    def _save_checkpoint(self):
        """Save a training checkpoint, used to resume interrupted training"""
        checkpoint = {
            'epoch': self.current_epoch + 1,  # Save the next epoch to be trained
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
        print(f"Checkpoint saved: {checkpoint_path}, current epoch: {self.current_epoch}")
    
    def _load_checkpoint(self):
        """Load a training checkpoint, used to resume interrupted training"""
        checkpoint_path = os.path.join('results', self.experiment_name, 'checkpoints', 'checkpoint.pt')
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load the model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load the optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load the training history
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
            self.best_val_acc = checkpoint['best_val_acc']
            
            # Set the starting epoch
            self.start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {self.start_epoch} onward")
        else:
            print("No checkpoint found, training from scratch")
    
    def train(self):
        """Full training process"""
        print(f"Starting training | Model: {type(self.model).__name__} | Device: {self.device}")
        print(f"Config: dataset={self.dataset_name}, optimizer={self.optimizer_name}, learning rate={self.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch()
            
            # Print statistics
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Save a checkpoint
            self._save_checkpoint()
            
            # Plot the current training curves
            self._plot_training_curves()
        
        # Save the final model
        self._save_model('final_model.pt')
        
        # Save the training history
        self._save_history()
        
        # Training finished
        train_time = time.time() - start_time
        print(f"Training complete! Total time: {train_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def _save_model(self, filename):
        """Save the model"""
        save_dir = os.path.join('results', self.experiment_name, 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, filename))
    
    def _save_history(self):
        """Save the training history"""
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
        """Plot the training curves"""
        save_dir = os.path.join('results', self.experiment_name, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        
        # Accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
        
        # Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
        
        plt.close('all')
    
    def test_model(self):
        """Evaluate the model on the test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        print(f"Test accuracy: {test_acc:.2f}%")
        return test_acc

# Config loader function
def load_config(config_path):
    """Load a configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 

# Add the VGG_A_BatchNorm model
class VGG_A_BatchNorm(nn.Module):
    """
    VGG_A model with BatchNorm
    
    Implements the same network architecture as VGG_A, but adds a BatchNorm2d layer after every convolution layer
    Inputs are assumed to be 32x32x3 CIFAR10 images
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
    Compute the model's accuracy on the given data loader
    
    Args:
        model: The model to evaluate
        data_loader: The data loader
        device: The compute device (CPU or CUDA)
        
    Returns:
        float: Accuracy as a percentage
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