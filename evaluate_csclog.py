#!/usr/bin/env python
"""
Evaluate trained CSCLog model on test set
Usage: python evaluate_csclog.py --model_path model/CSCLog/CSCLog.pt --data_dir dataset/processed
"""

import argparse
import torch
import pandas as pd
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Import from train_csclog
import sys
sys.path.append('.')
from train_csclog import Model, generate_pre


def evaluate_model(model_path, data_dir, window_size=9, device='cuda'):
    """Evaluate trained model on test set"""
    
    print("="*80)
    print("CSCLog Model Evaluation")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load test data
    test_path = f"{data_dir}/test_normal.csv"
    temp_path = f"{data_dir}/log_templates.csv"
    emb_path = f"{data_dir}/sentences_emb.json"
    com_path = f"{data_dir}/component.json"
    
    print("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Total test sequences: {len(test_df)}")
    print(f"Normal sequences (Label=0): {len(test_df[test_df['Label']==0])}")
    print(f"Anomaly sequences (Label=1): {len(test_df[test_df['Label']==1])}")
    
    if len(test_df[test_df['Label']==1]) == 0:
        print("\n⚠️  Warning: No anomaly sequences found in test set!")
        print("Cannot evaluate anomaly detection performance.")
        return
    
    # Load model metadata
    logTemp = pd.read_csv(temp_path, index_col='EventId')
    emb = json.load(open(emb_path, 'r'))
    cop = json.load(open(com_path, 'r'))
    
    attr_num = len(list(emb.items())[0][1])
    class_num = len(logTemp.index.unique())
    com_num = len(cop)
    
    # Create model
    print("\nLoading model...")
    hidden_size = [64, 64, 64, 64, 64]
    model = Model(attr_num, com_num, hidden_size, 0.8, 1, 2, class_num, 0.1, True).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Generate predictions
    print("\nGenerating predictions...")
    test_loader, _, _ = generate_pre('test', test_path, temp_path, emb_path, com_path, window_size)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for seq, com, quan, timp, label in test_loader:
            seq = seq.to(device)
            com = com.to(device)
            quan = quan.to(device)
            timp = timp.to(device)
            
            output = model(seq, com, quan, timp)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(label.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:7d}")
    print(f"      Anomaly   {cm[1][0]:6d}  {cm[1][1]:7d}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['Normal', 'Anomaly']))
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels),
        'normal_samples': int(np.sum(all_labels == 0)),
        'anomaly_samples': int(np.sum(all_labels == 1))
    }
    
    results_path = f"{data_dir}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSCLog model')
    parser.add_argument('--model_path', type=str, default='model/CSCLog/CSCLog.pt',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='dataset/processed',
                       help='Directory containing processed data')
    parser.add_argument('--window_size', type=int, default=9,
                       help='Window size used during training')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_model(args.model_path, args.data_dir, args.window_size, device)


if __name__ == '__main__':
    main()
