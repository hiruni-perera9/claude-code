"""
Evaluation script for anomaly detection models
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report, roc_curve
)
from typing import Dict, Tuple
import json
import os

from models import get_model, TabularDataset
from torch.utils.data import DataLoader


class AnomalyDetectionEvaluator:
    """Evaluator for trained anomaly detection models"""

    def __init__(self,
                 model: torch.nn.Module,
                 checkpoint_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Checkpoint val_loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores

        Returns:
            anomaly_scores, labels (if available)
        """
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    inputs, labels = batch
                    all_labels.extend(labels.cpu().numpy())
                else:
                    inputs = batch
                    labels = None

                inputs = inputs.to(self.device)

                # Forward pass
                if hasattr(self.model, 'model1'):  # Ensemble model
                    reconstruction, _, _ = self.model(inputs)
                    scores = self.model.compute_anomaly_score(inputs, reconstruction)
                else:
                    reconstruction, _ = self.model(inputs)
                    scores = self.model.compute_anomaly_score(inputs, reconstruction)

                all_scores.extend(scores.cpu().numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels) if len(all_labels) > 0 else None

        return all_scores, all_labels

    def find_optimal_threshold(self,
                               scores: np.ndarray,
                               labels: np.ndarray) -> float:
        """Find optimal threshold using Youden's J statistic"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"  TPR: {tpr[optimal_idx]:.4f}")
        print(f"  FPR: {fpr[optimal_idx]:.4f}")

        return optimal_threshold

    def evaluate(self,
                 dataloader: DataLoader,
                 threshold: float = None) -> Dict:
        """
        Comprehensive evaluation

        Args:
            dataloader: Data loader with labels
            threshold: Anomaly threshold (if None, will be auto-determined)

        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)

        # Get predictions
        scores, labels = self.predict(dataloader)

        if labels is None:
            raise ValueError("Labels required for evaluation")

        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self.find_optimal_threshold(scores, labels)

        # Binary predictions
        predictions = (scores > threshold).astype(int)

        # Compute metrics
        metrics = {
            'threshold': threshold,
            'roc_auc': roc_auc_score(labels, scores),
        }

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics['pr_auc'] = auc(recall, precision)

        # Classification metrics
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall'] /
                               (metrics['precision'] + metrics['recall'])
                               if (metrics['precision'] + metrics['recall']) > 0 else 0)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }

        # Print results
        print("\nMetrics:")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

        print("\nConfusion Matrix:")
        print(f"  TN: {tn:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TP: {tp:5d}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(labels, predictions,
                                   target_names=['Normal', 'Anomaly']))

        return metrics, scores, labels, predictions

    def plot_evaluation(self,
                       scores: np.ndarray,
                       labels: np.ndarray,
                       predictions: np.ndarray,
                       threshold: float,
                       save_path: str = './evaluation_plots.png'):
        """Generate evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = roc_auc_score(labels, scores)

        axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        axes[0, 1].plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. Score Distribution
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        axes[1, 0].hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
        axes[1, 0].hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', density=True)
        axes[1, 0].axvline(threshold, color='r', linestyle='--', label='Threshold')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xticklabels(['Normal', 'Anomaly'])
        axes[1, 1].set_yticklabels(['Normal', 'Anomaly'])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEvaluation plots saved to {save_path}")


def main():
    """Main evaluation function"""
    # Configuration
    checkpoint_dir = './checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Load configuration and metadata
    with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(checkpoint_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    print("\n" + "="*60)
    print("PALEODB ANOMALY DETECTION - EVALUATION")
    print("="*60)

    # Prepare test data (using same data preparation as training)
    from train import prepare_data

    _, val_loader, _ = prepare_data(
        limit=config['data_limit'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        anomaly_ratio=config['anomaly_ratio']
    )

    # Initialize model
    print(f"\nLoading {config['model_type']} model...")
    model = get_model(
        model_type=config['model_type'],
        input_dim=metadata['input_dim'],
        dropout=0.1
    )

    # Initialize evaluator
    evaluator = AnomalyDetectionEvaluator(
        model=model,
        checkpoint_path=checkpoint_path
    )

    # Evaluate
    metrics, scores, labels, predictions = evaluator.evaluate(val_loader)

    # Plot results
    evaluator.plot_evaluation(
        scores, labels, predictions,
        threshold=metrics['threshold'],
        save_path='./evaluation_plots.png'
    )

    # Save metrics
    with open('./evaluation_metrics.json', 'w') as f:
        # Convert numpy types to native Python types
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_serializable, f, indent=2)

    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
