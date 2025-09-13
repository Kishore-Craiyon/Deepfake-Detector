import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ensemble_model import DeepFakeEnsemble
from training.utils import MetricsTracker, save_checkpoint, plot_training_history
import wandb

class DeepFakeTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepFakeEnsemble(self.config).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        
        # Initialize wandb for logging
        wandb.init(project="deepfake-detection", config=self.config)
        
    def train_epoch(self, train_loader):
        self.model.train()
        metrics = MetricsTracker()
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_data, targets in progress_bar:
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_data)
            loss = self.criterion(outputs['final_output'], targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                preds = torch.softmax(outputs['final_output'], dim=1)
                metrics.update(preds, targets, loss.item())
                
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        return metrics.compute_metrics()
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch_data, targets in progress_bar:
                # Move data to device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs['final_output'], targets)
                
                # Update metrics
                preds = torch.softmax(outputs['final_output'], dim=1)
                metrics.update(preds, targets, loss.item())
                
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
        return metrics.compute_metrics()
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['loss'],
                    'models/best_model.pth'
                )
                print("✓ Saved best model")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
        # Plot training history
        plot_training_history(
            history['train_loss'], history['val_loss'],
            history['train_acc'], history['val_acc']
        )
        
        return history