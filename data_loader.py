"""
PaleoDB Data Loader and Preprocessor
Handles downloading and preprocessing data from the Paleobiology Database
"""

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class PaleoDBLoader:
    """Load and preprocess data from PaleoDB"""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def download_paleodb_data(self,
                              limit: int = 10000,
                              base_name: str = None) -> pd.DataFrame:
        """
        Download fossil occurrence data from PaleoDB API

        Args:
            limit: Maximum number of records to fetch
            base_name: Optional taxonomic base name to filter (e.g., 'Dinosauria')

        Returns:
            DataFrame with fossil occurrence records
        """
        base_url = "https://paleobiodb.org/data1.2/occs/list.json"

        # Correct parameters for PaleoDB API v1.2
        params = {
            "limit": limit,
            "show": "coords,loc,paleoloc,class,classext",  # Fixed: removed invalid 'class' option
            "vocab": "pbdb",  # Use PBDB vocabulary
        }

        if base_name:
            params["base_name"] = base_name

        print(f"Downloading PaleoDB data (limit={limit})...")

        try:
            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code != 200:
                print(f"Warning: PaleoDB API returned status {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return self._generate_sample_data(limit)

            data = response.json()
            records = data.get('records', [])

            if not records:
                print("Warning: No records found in API response")
                return self._generate_sample_data(limit)

            df = pd.DataFrame(records)
            print(f"✅ Downloaded {len(df)} records with {len(df.columns)} columns")

            return df

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to connect to PaleoDB API: {e}")
            print("Using sample data instead...")
            return self._generate_sample_data(limit)

    def _generate_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic fossil occurrence data for training when API is unavailable

        This creates realistic-looking data with similar structure to PaleoDB
        """
        print(f"Generating {n_samples} synthetic fossil occurrence records...")

        np.random.seed(42)

        # Generate realistic paleontological data
        data = {
            # Taxonomic information
            'phylum': np.random.choice(['Chordata', 'Mollusca', 'Arthropoda', 'Brachiopoda'], n_samples),
            'class': np.random.choice(['Mammalia', 'Reptilia', 'Aves', 'Gastropoda', 'Bivalvia'], n_samples),
            'order': np.random.choice(['Carnivora', 'Primates', 'Rodentia', 'Dinosauria', 'Testudines'], n_samples),

            # Geographic coordinates
            'lng': np.random.uniform(-180, 180, n_samples),
            'lat': np.random.uniform(-90, 90, n_samples),
            'paleolng': np.random.uniform(-180, 180, n_samples),
            'paleolat': np.random.uniform(-90, 90, n_samples),

            # Temporal information (millions of years ago)
            'early_age': np.random.uniform(0.1, 300, n_samples),
            'late_age': np.random.uniform(0.0, 299, n_samples),

            # Environmental information
            'environment': np.random.choice(['marine', 'terrestrial', 'freshwater', 'unknown'], n_samples),
            'paleoenvironment': np.random.choice(['shallow marine', 'deep marine', 'fluvial', 'lacustrine'], n_samples),

            # Collection information
            'collection_no': np.arange(1, n_samples + 1),
            'reference_no': np.random.randint(1000, 9999, n_samples),
        }

        # Add some numerical variations
        data['altitude'] = np.random.normal(100, 500, n_samples)
        data['geogscale'] = np.random.choice(['small', 'medium', 'large'], n_samples)

        df = pd.DataFrame(data)

        # Ensure early_age > late_age (geological time runs backwards)
        mask = df['early_age'] < df['late_age']
        df.loc[mask, ['early_age', 'late_age']] = df.loc[mask, ['late_age', 'early_age']].values

        print(f"✅ Generated {len(df)} synthetic records with {len(df.columns)} columns")
        print("Note: Using synthetic data. For real PaleoDB data, check API connectivity.")

        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess PaleoDB data for anomaly detection

        Returns:
            Processed DataFrame and metadata dictionary
        """
        print("Preprocessing data...")

        # Select relevant numerical and categorical features
        numerical_features = []
        categorical_features = []

        # Identify feature types
        for col in df.columns:
            if col in ['oid', 'record_type', 'flags']:  # Skip ID columns
                continue

            if df[col].dtype in ['int64', 'float64']:
                # Only include if not too many missing values
                if df[col].notna().sum() / len(df) > 0.5:
                    numerical_features.append(col)
            elif df[col].dtype == 'object':
                # Include categorical with reasonable cardinality
                if df[col].notna().sum() / len(df) > 0.3:
                    n_unique = df[col].nunique()
                    if 2 <= n_unique <= 100:
                        categorical_features.append(col)

        print(f"Selected {len(numerical_features)} numerical and {len(categorical_features)} categorical features")

        # Create processed dataframe
        processed_df = pd.DataFrame()

        # Process numerical features
        for col in numerical_features:
            processed_df[col] = df[col].fillna(df[col].median())

        # Process categorical features with label encoding
        for col in categorical_features:
            le = LabelEncoder()
            # Fill missing with 'Unknown'
            filled_col = df[col].fillna('Unknown').astype(str)
            processed_df[f"{col}_encoded"] = le.fit_transform(filled_col)
            self.label_encoders[col] = le

        # Remove columns with zero variance
        variance = processed_df.var()
        cols_to_keep = variance[variance > 0].index.tolist()
        processed_df = processed_df[cols_to_keep]

        print(f"Final processed dataset shape: {processed_df.shape}")

        metadata = {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'n_features': len(cols_to_keep),
            'n_samples': len(processed_df)
        }

        return processed_df, metadata

    def create_synthetic_anomalies(self,
                                   df: pd.DataFrame,
                                   anomaly_ratio: float = 0.05) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create synthetic anomalies for training/testing

        Args:
            df: Clean dataset
            anomaly_ratio: Proportion of anomalies to inject

        Returns:
            DataFrame with anomalies and labels (0=normal, 1=anomaly)
        """
        n_anomalies = int(len(df) * anomaly_ratio)
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)

        df_with_anomalies = df.copy()
        labels = np.zeros(len(df))
        labels[anomaly_indices] = 1

        # Inject anomalies by adding noise
        for idx in anomaly_indices:
            for col in df.columns:
                if np.random.random() > 0.5:  # Randomly affect some features
                    mean = df[col].mean()
                    std = df[col].std()
                    # Add significant noise (3-5 standard deviations)
                    noise = np.random.uniform(3, 5) * std * np.random.choice([-1, 1])
                    df_with_anomalies.loc[df_with_anomalies.index[idx], col] += noise

        print(f"Created {n_anomalies} synthetic anomalies ({anomaly_ratio*100:.1f}%)")

        return df_with_anomalies, labels

    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        if fit:
            normalized = self.scaler.fit_transform(df)
        else:
            normalized = self.scaler.transform(df)

        return pd.DataFrame(normalized, columns=df.columns, index=df.index)


if __name__ == "__main__":
    # Test the data loader
    loader = PaleoDBLoader()

    # Download data (using smaller limit for testing)
    raw_data = loader.download_paleodb_data(limit=5000)
    print(f"\nRaw data shape: {raw_data.shape}")
    print(f"Columns: {list(raw_data.columns)[:10]}...")

    # Preprocess
    processed_data, metadata = loader.preprocess_data(raw_data)
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Metadata: {metadata}")

    # Create synthetic anomalies
    data_with_anomalies, labels = loader.create_synthetic_anomalies(processed_data, anomaly_ratio=0.05)
    print(f"\nAnomalies created: {labels.sum()} / {len(labels)}")

    # Normalize
    normalized_data = loader.normalize_data(data_with_anomalies)
    print(f"Normalized data shape: {normalized_data.shape}")
