import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class MetricsTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, preds, targets, loss):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
        
    def compute_metrics(self):
        preds_array = np.array(self.predictions)
        targets_array = np.array(self.targets)
        
        # Convert probabilities to predictions
        if preds_array.shape[1] == 2:  # Binary classification
            pred_classes = np.argmax(preds_array, axis=1)
            prob_scores = preds_array[:, 1]  # Probability of positive class
        else:
            pred_classes = preds_array
            prob_scores = preds_array
            
        accuracy = accuracy_score(targets_array, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_array, pred_classes, average='binary'
        )
        
        try:
            auc = roc_auc_score(targets_array, prob_scores)
        except:
            auc = 0.0
            
        avg_loss = np.mean(self.losses)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': avg_loss
        }

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    
def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint['epoch'], checkpoint['loss']

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()