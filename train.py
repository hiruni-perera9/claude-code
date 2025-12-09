"""
Training pipeline for anomaly detection models on PaleoDB
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Tuple, Dict, Optional
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from models import get_model, TabularDataset
from data_loader import PaleoDBLoader


class AnomalyDetectionTrainer:
    """Trainer class for anomaly detection models"""

    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Loss function
        self.criterion = nn.MSELoss(reduction='none')

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            if isinstance(batch, tuple):
                inputs, _ = batch
            else:
                inputs = batch

            # Ensure inputs is a tensor
            if not torch.is_tensor(inputs):
                if isinstance(inputs, (list, np.ndarray)):
                    inputs = torch.FloatTensor(inputs)
                else:
                    raise TypeError(f"Unexpected input type: {type(inputs)}")

            inputs = inputs.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if hasattr(self.model, 'model1'):  # Ensemble model
                reconstruction, _, (recon1, recon2) = self.model(inputs)
            else:
                reconstruction, _ = self.model(inputs)

            # Compute loss
            loss = self.criterion(reconstruction, inputs).mean()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                if isinstance(batch, tuple):
                    inputs, labels = batch
                    all_labels.extend(labels.cpu().numpy())
                else:
                    inputs = batch
                    labels = None

                # Ensure inputs is a tensor
                if not torch.is_tensor(inputs):
                    if isinstance(inputs, (list, np.ndarray)):
                        inputs = torch.FloatTensor(inputs)
                    else:
                        raise TypeError(f"Unexpected input type: {type(inputs)}")

                inputs = inputs.to(self.device)

                # Forward pass
                if hasattr(self.model, 'model1'):  # Ensemble model
                    reconstruction, _, (recon1, recon2) = self.model(inputs)
                    scores = self.model.compute_anomaly_score(inputs, reconstruction)
                else:
                    reconstruction, _ = self.model(inputs)
                    scores = self.model.compute_anomaly_score(inputs, reconstruction)

                # Compute loss
                loss = self.criterion(reconstruction, inputs).mean()
                total_loss += loss.item()
                num_batches += 1

                all_scores.extend(scores.cpu().numpy())

        avg_loss = total_loss / num_batches

        # Compute metrics if labels are available
        metrics = {}
        if len(all_labels) > 0:
            all_labels = np.array(all_labels)
            all_scores = np.array(all_scores)

            # ROC-AUC
            metrics['roc_auc'] = roc_auc_score(all_labels, all_scores)

            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            metrics['pr_auc'] = auc(recall, precision)

        return avg_loss, metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              early_stopping_patience: int = 10,
              checkpoint_dir: str = './checkpoints') -> Dict:
        """
        Full training loop with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            if metrics:
                print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

        # Load best model
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

        return self.history

    def plot_training_history(self, save_path: str = './training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Learning rate plot
        axes[1].plot(self.history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")


def prepare_data(limit: int = 10000,
                 batch_size: int = 64,
                 val_split: float = 0.2,
                 anomaly_ratio: float = 0.05) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Prepare PaleoDB data for training

    Returns:
        train_loader, val_loader, metadata
    """
    print("="*60)
    print("Preparing PaleoDB data...")
    print("="*60)

    # Load data
    loader = PaleoDBLoader()
    raw_data = loader.download_paleodb_data(limit=limit)

    # Preprocess
    processed_data, metadata = loader.preprocess_data(raw_data)

    # Create synthetic anomalies
    data_with_anomalies, labels = loader.create_synthetic_anomalies(
        processed_data, anomaly_ratio=anomaly_ratio
    )

    # Normalize
    normalized_data = loader.normalize_data(data_with_anomalies)

    # Convert to numpy
    X = normalized_data.values
    y = labels

    # Create dataset
    dataset = TabularDataset(X, y)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\nData prepared:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Features:      {X.shape[1]}")
    print(f"  Anomaly ratio: {anomaly_ratio*100:.1f}%")

    metadata['input_dim'] = X.shape[1]

    return train_loader, val_loader, metadata


def main():
    """Main training function"""
    # Configuration
    config = {
        'data_limit': 10000,
        'batch_size': 64,
        'val_split': 0.2,
        'anomaly_ratio': 0.05,
        'model_type': 'huggingface',  # 'transformer', 'huggingface', or 'ensemble'
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'early_stopping_patience': 10,
        'checkpoint_dir': './checkpoints'
    }

    print("\n" + "="*60)
    print("PALEODB ANOMALY DETECTION - TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Prepare data
    train_loader, val_loader, metadata = prepare_data(
        limit=config['data_limit'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        anomaly_ratio=config['anomaly_ratio']
    )

    # Initialize model
    print(f"\nInitializing {config['model_type']} model...")
    model = get_model(
        model_type=config['model_type'],
        input_dim=metadata['input_dim'],
        dropout=0.1
    )

    # Initialize trainer
    trainer = AnomalyDetectionTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_dir=config['checkpoint_dir']
    )

    # Plot training history
    trainer.plot_training_history('./training_history.png')

    # Save configuration and metadata
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(config['checkpoint_dir'], 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
