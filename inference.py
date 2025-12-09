"""
Inference script for anomaly detection on new PaleoDB data
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict
import argparse

from models import get_model
from data_loader import PaleoDBLoader


class AnomalyDetector:
    """Production-ready anomaly detector"""

    def __init__(self,
                 checkpoint_dir: str = './checkpoints',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Load configuration
        with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)

        with open(os.path.join(checkpoint_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)

        # Load model
        self.model = get_model(
            model_type=self.config['model_type'],
            input_dim=self.metadata['input_dim'],
            dropout=0.0  # No dropout for inference
        ).to(device)

        # Load weights
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load threshold from evaluation metrics
        metrics_path = './evaluation_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                self.threshold = metrics['threshold']
        else:
            self.threshold = 0.5  # Default threshold

        # Initialize data loader
        self.data_loader = PaleoDBLoader()

        print(f"Anomaly Detector initialized")
        print(f"  Model: {self.config['model_type']}")
        print(f"  Device: {self.device}")
        print(f"  Threshold: {self.threshold:.4f}")

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess new data using the same pipeline as training"""
        processed_data, _ = self.data_loader.preprocess_data(data)

        # Ensure same feature dimensionality
        if processed_data.shape[1] != self.metadata['input_dim']:
            print(f"Warning: Feature mismatch. Expected {self.metadata['input_dim']}, got {processed_data.shape[1]}")

            # Pad or truncate to match expected dimensions
            if processed_data.shape[1] < self.metadata['input_dim']:
                # Pad with zeros
                padding = np.zeros((processed_data.shape[0],
                                   self.metadata['input_dim'] - processed_data.shape[1]))
                processed_data = pd.DataFrame(
                    np.hstack([processed_data.values, padding])
                )
            else:
                # Truncate
                processed_data = processed_data.iloc[:, :self.metadata['input_dim']]

        # Normalize using training statistics
        normalized_data = self.data_loader.normalize_data(processed_data, fit=False)

        return normalized_data.values

    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Predict anomalies in new data

        Args:
            data: DataFrame with PaleoDB fossil occurrence records

        Returns:
            Dictionary containing:
                - anomaly_scores: Anomaly score for each sample
                - is_anomaly: Binary predictions
                - anomaly_indices: Indices of detected anomalies
                - anomaly_probability: Probability-like scores (normalized)
        """
        # Preprocess
        X = self.preprocess_data(data)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        with torch.no_grad():
            if hasattr(self.model, 'model1'):  # Ensemble
                reconstruction, _, _ = self.model(X_tensor)
                scores = self.model.compute_anomaly_score(X_tensor, reconstruction)
            else:
                reconstruction, _ = self.model(X_tensor)
                scores = self.model.compute_anomaly_score(X_tensor, reconstruction)

        scores = scores.cpu().numpy()

        # Binary predictions
        predictions = (scores > self.threshold).astype(int)

        # Get anomaly indices
        anomaly_indices = np.where(predictions == 1)[0].tolist()

        # Normalize scores to probability-like range [0, 1]
        # Using sigmoid-like transformation
        prob_scores = 1 / (1 + np.exp(-10 * (scores - self.threshold)))

        results = {
            'anomaly_scores': scores.tolist(),
            'is_anomaly': predictions.tolist(),
            'anomaly_indices': anomaly_indices,
            'anomaly_probability': prob_scores.tolist(),
            'n_anomalies': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(data)
        }

        return results

    def predict_and_explain(self, data: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Predict anomalies and provide detailed results

        Args:
            data: Input DataFrame
            top_k: Return top K most anomalous samples

        Returns:
            DataFrame with anomaly scores and original data
        """
        results = self.predict(data)

        # Create results dataframe
        results_df = data.copy()
        results_df['anomaly_score'] = results['anomaly_scores']
        results_df['is_anomaly'] = results['is_anomaly']
        results_df['anomaly_probability'] = results['anomaly_probability']

        # Sort by anomaly score
        results_df = results_df.sort_values('anomaly_score', ascending=False)

        print(f"\nAnomaly Detection Results:")
        print(f"  Total samples: {len(data)}")
        print(f"  Anomalies detected: {results['n_anomalies']}")
        print(f"  Anomaly rate: {results['anomaly_rate']*100:.2f}%")

        if results['n_anomalies'] > 0:
            print(f"\nTop {min(top_k, results['n_anomalies'])} most anomalous samples:")
            print(results_df.head(top_k)[['anomaly_score', 'anomaly_probability', 'is_anomaly']])

        return results_df


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Detect anomalies in PaleoDB data')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Number of records to download from PaleoDB')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--output', type=str, default='./anomaly_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top anomalies to display')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PALEODB ANOMALY DETECTION - INFERENCE")
    print("="*60)

    # Initialize detector
    detector = AnomalyDetector(checkpoint_dir=args.checkpoint_dir)

    # Download new data
    print(f"\nDownloading {args.limit} records from PaleoDB...")
    loader = PaleoDBLoader()
    new_data = loader.download_paleodb_data(limit=args.limit)

    # Predict anomalies
    print("\nDetecting anomalies...")
    results_df = detector.predict_and_explain(new_data, top_k=args.top_k)

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Save summary
    summary = {
        'total_samples': len(results_df),
        'n_anomalies': int(results_df['is_anomaly'].sum()),
        'anomaly_rate': float(results_df['is_anomaly'].mean()),
        'mean_anomaly_score': float(results_df['anomaly_score'].mean()),
        'max_anomaly_score': float(results_df['anomaly_score'].max()),
        'min_anomaly_score': float(results_df['anomaly_score'].min())
    }

    summary_path = args.output.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")

    print("\n" + "="*60)
    print("Inference completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
