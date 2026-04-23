#!/usr/bin/env python
"""
Sample training data to reduce training time
Usage: python sample_training_data.py --sample_size 50000
"""

import argparse
import pandas as pd
import os


def sample_data(input_file, output_file, sample_size, keep_anomaly_ratio=True):
    """Sample training data while maintaining anomaly ratio"""
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    total = len(df)
    normal = len(df[df['Label'] == 0])
    anomaly = len(df[df['Label'] == 1])
    
    print(f"Original data: {total} sequences ({normal} normal, {anomaly} anomaly)")
    
    if total <= sample_size:
        print(f"Data already smaller than {sample_size}, copying as-is...")
        df.to_csv(output_file, index=False)
        return
    
    if keep_anomaly_ratio and anomaly > 0:
        # Calculate how many of each to sample
        anomaly_ratio = anomaly / total
        sample_anomaly = int(sample_size * anomaly_ratio)
        sample_normal = sample_size - sample_anomaly
        
        # Sample separately
        df_normal = df[df['Label'] == 0].sample(n=min(sample_normal, normal), random_state=42)
        df_anomaly = df[df['Label'] == 1].sample(n=min(sample_anomaly, anomaly), random_state=42)
        
        # Combine and shuffle
        df_sampled = pd.concat([df_normal, df_anomaly]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Sampled: {len(df_sampled)} sequences ({len(df_normal)} normal, {len(df_anomaly)} anomaly)")
    else:
        # Simple random sample
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"Sampled: {len(df_sampled)} sequences")
    
    # Save
    df_sampled.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Sample training data')
    parser.add_argument('--data_dir', type=str, default='dataset/processed',
                       help='Data directory')
    parser.add_argument('--sample_size', type=int, default=50000,
                       help='Number of sequences to sample')
    parser.add_argument('--keep_ratio', action='store_true', default=True,
                       help='Keep anomaly ratio')
    
    args = parser.parse_args()
    
    # Backup original
    train_file = os.path.join(args.data_dir, 'train_normal.csv')
    backup_file = os.path.join(args.data_dir, 'train_normal_full.csv')
    
    if not os.path.exists(backup_file):
        print(f"Backing up original to {backup_file}...")
        import shutil
        shutil.copy(train_file, backup_file)
    
    # Sample
    sample_data(backup_file, train_file, args.sample_size, args.keep_ratio)
    
    print("\n✅ Done! Now you can train faster:")
    print(f"   python train_csclog.py --batch_size 128 --num_epochs 20")
    print(f"\nTo restore full data:")
    print(f"   cp {backup_file} {train_file}")


if __name__ == '__main__':
    main()
