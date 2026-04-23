"""
Data Splitter for CSCLog
Splits processed sequences into training, validation, and test sets
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Split sequences into train/val/test sets"""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, random_seed: int = 42):
        """
        Initialize splitter with ratios and random seed
        
        Args:
            train_ratio: Ratio for training set (normal data only)
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    def split(self, sequences: pd.DataFrame, stratify_by: str = "Label") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split sequences into train/val/test sets
        
        Strategy:
        - Normal sequences: 70% train, 15% val, 15% test
        - Anomaly sequences: 0% train, 50% val, 50% test
        
        Args:
            sequences: DataFrame with SessionId, EventSequence, Label
            stratify_by: Column to stratify by (default: Label)
            
        Returns:
            train_df, val_df, test_df
        """
        print(f"Splitting {len(sequences)} sequences...")
        
        # Separate normal and anomaly sequences
        normal_seqs = sequences[sequences['Label'] == 0].copy()
        anomaly_seqs = sequences[sequences['Label'] == 1].copy()
        
        print(f"Normal sequences: {len(normal_seqs)}")
        print(f"Anomaly sequences: {len(anomaly_seqs)}")
        
        # Split normal sequences: 80% train, 20% test (or custom ratios)
        if len(normal_seqs) > 0:
            # If val_ratio is 0, do simple train/test split
            if self.val_ratio == 0:
                train_normal, test_normal = train_test_split(
                    normal_seqs,
                    test_size=self.test_ratio,
                    random_state=self.random_seed
                )
                val_normal = pd.DataFrame(columns=sequences.columns)
            else:
                # First split: train vs (val+test)
                train_normal, temp_normal = train_test_split(
                    normal_seqs,
                    test_size=(self.val_ratio + self.test_ratio),
                    random_state=self.random_seed
                )
                
                # Second split: val vs test
                val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
                val_normal, test_normal = train_test_split(
                    temp_normal,
                    test_size=(1 - val_ratio_adjusted),
                    random_state=self.random_seed
                )
        else:
            train_normal = pd.DataFrame(columns=sequences.columns)
            val_normal = pd.DataFrame(columns=sequences.columns)
            test_normal = pd.DataFrame(columns=sequences.columns)
        
        # Split anomaly sequences: 0% train, rest goes to val/test
        if len(anomaly_seqs) > 0:
            if self.val_ratio == 0:
                # No validation set, all anomalies go to test
                val_anomaly = pd.DataFrame(columns=sequences.columns)
                test_anomaly = anomaly_seqs
            else:
                # Split anomalies between val and test (50/50)
                val_anomaly, test_anomaly = train_test_split(
                    anomaly_seqs,
                    test_size=0.5,
                    random_state=self.random_seed
                )
        else:
            val_anomaly = pd.DataFrame(columns=sequences.columns)
            test_anomaly = pd.DataFrame(columns=sequences.columns)
        
        # Combine splits
        train_df = train_normal
        val_df = pd.concat([val_normal, val_anomaly], ignore_index=True)
        test_df = pd.concat([test_normal, test_anomaly], ignore_index=True)
        
        # Shuffle
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"\nSplit results:")
        print(f"  Train: {len(train_df)} sequences (normal only)")
        print(f"  Val:   {len(val_df)} sequences ({len(val_normal)} normal, {len(val_anomaly)} anomaly)")
        print(f"  Test:  {len(test_df)} sequences ({len(test_normal)} normal, {len(test_anomaly)} anomaly)")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, output_dir: str):
        """
        Save splits to separate CSV files
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Directory to save files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train (normal only)
        train_path = os.path.join(output_dir, "train_normal.csv")
        train_df.to_csv(train_path, index=False)
        print(f"Saved training set to {train_path}")
        
        # Save validation (split by label)
        val_normal = val_df[val_df['Label'] == 0]
        val_anomaly = val_df[val_df['Label'] == 1]
        
        if len(val_normal) > 0:
            val_normal_path = os.path.join(output_dir, "val_normal.csv")
            val_normal.to_csv(val_normal_path, index=False)
            print(f"Saved validation normal set to {val_normal_path}")
        
        if len(val_anomaly) > 0:
            val_anomaly_path = os.path.join(output_dir, "val_anomaly.csv")
            val_anomaly.to_csv(val_anomaly_path, index=False)
            print(f"Saved validation anomaly set to {val_anomaly_path}")
        
        # Save test (split by label)
        test_normal = test_df[test_df['Label'] == 0]
        test_anomaly = test_df[test_df['Label'] == 1]
        
        if len(test_normal) > 0:
            test_normal_path = os.path.join(output_dir, "test_normal.csv")
            test_normal.to_csv(test_normal_path, index=False)
            print(f"Saved test normal set to {test_normal_path}")
        
        if len(test_anomaly) > 0:
            test_anomaly_path = os.path.join(output_dir, "test_anomaly.csv")
            test_anomaly.to_csv(test_anomaly_path, index=False)
            print(f"Saved test anomaly set to {test_anomaly_path}")
        
        print("\nAll splits saved successfully!")
