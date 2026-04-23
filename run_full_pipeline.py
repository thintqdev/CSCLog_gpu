#!/usr/bin/env python
"""
CSCLog Full Pipeline - Preprocessing + Training
Run complete pipeline from raw logs to trained model
Usage: python run_full_pipeline.py [--skip-preprocessing] [--skip-training]
"""

import argparse
import sys
import os
import subprocess
import time


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Command: {cmd}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='CSCLog Full Pipeline - Preprocessing + Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_full_pipeline.py
  
  # Skip preprocessing (if already done)
  python run_full_pipeline.py --skip-preprocessing
  
  # Only run preprocessing
  python run_full_pipeline.py --skip-training
  
  # Custom config and parameters
  python run_full_pipeline.py --config my_config.yaml --epochs 5
        """
    )
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file for preprocessing')
    parser.add_argument('--data-dir', type=str, default='dataset/processed',
                       help='Directory for processed data')
    parser.add_argument('--model-dir', type=str, default='model/CSCLog',
                       help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--window-size', type=int, default=9,
                       help='Window size')
    
    args = parser.parse_args()
    
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                           ║")
    print("║   CSCLog Full Pipeline                                                   ║")
    print("║   Preprocessing + Training                                               ║")
    print("║                                                                           ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    start_time = time.time()
    
    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        print("\n📊 Step 1/2: Data Preprocessing")
        
        # Check if preprocessing already done
        required_files = [
            os.path.join(args.data_dir, 'log_templates.csv'),
            os.path.join(args.data_dir, 'sentences_emb.json'),
            os.path.join(args.data_dir, 'component.json'),
            os.path.join(args.data_dir, 'train_normal.csv'),
        ]
        
        all_exist = all(os.path.exists(f) for f in required_files)
        
        if all_exist:
            print("\n⚠️  Preprocessed data already exists!")
            response = input("Do you want to re-run preprocessing? (y/N): ")
            if response.lower() != 'y':
                print("Skipping preprocessing...")
            else:
                cmd = f"python run_preprocessing.py --config {args.config}"
                if not run_command(cmd, "Data Preprocessing"):
                    return 1
        else:
            cmd = f"python run_preprocessing.py --config {args.config}"
            if not run_command(cmd, "Data Preprocessing"):
                return 1
    else:
        print("\n⏭️  Skipping preprocessing (--skip-preprocessing)")
    
    # Step 2: Training
    if not args.skip_training:
        print("\n🚀 Step 2/3: Model Training")
        
        # Check if preprocessed data exists
        required_files = [
            os.path.join(args.data_dir, 'log_templates.csv'),
            os.path.join(args.data_dir, 'sentences_emb.json'),
            os.path.join(args.data_dir, 'component.json'),
            os.path.join(args.data_dir, 'train_normal.csv'),
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print("\n❌ Error: Preprocessed data not found!")
            print("Missing files:")
            for f in missing:
                print(f"  - {f}")
            print("\nPlease run preprocessing first or remove --skip-preprocessing")
            return 1
        
        cmd = (f"python train_csclog.py "
               f"--data_dir {args.data_dir} "
               f"--model_dir {args.model_dir} "
               f"--num_epochs {args.epochs} "
               f"--batch_size {args.batch_size} "
               f"--lr {args.lr} "
               f"--window_size {args.window_size}")
        
        if not run_command(cmd, "Model Training"):
            return 1
    else:
        print("\n⏭️  Skipping training (--skip-training)")
    
    # Step 3: Evaluation
    if not args.skip_training:
        print("\n📊 Step 3/3: Model Evaluation")
        
        model_path = os.path.join(args.model_dir, 'CSCLog.pt')
        if not os.path.exists(model_path):
            print(f"\n⚠️  Warning: Model not found at {model_path}")
            print("Skipping evaluation...")
        else:
            cmd = (f"python evaluate.py "
                   f"--model_path {model_path} "
                   f"--data_dir {args.data_dir} "
                   f"--window_size {args.window_size}")
            
            if not run_command(cmd, "Model Evaluation"):
                print("\n⚠️  Evaluation failed, but training completed successfully")
    else:
        print("\n⏭️  Skipping evaluation (--skip-training)")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if not args.skip_preprocessing:
        print(f"\n📊 Preprocessed data: {args.data_dir}/")
        print("   - log_templates.csv")
        print("   - sentences_emb.json")
        print("   - component.json")
        print("   - train_normal.csv")
        print("   - test_normal.csv")
        print("   - test_anomaly.csv (if anomalies exist)")
    
    if not args.skip_training:
        print(f"\n🤖 Trained model: {args.model_dir}/CSCLog.pt")
        print(f"📊 Evaluation results: {args.data_dir}/evaluation_results.json")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
