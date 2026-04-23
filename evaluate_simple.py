#!/usr/bin/env python
"""
Simple evaluation - classify sequences directly without generate_pre
Usage: python evaluate_simple.py --model_path model/CSCLog/CSCLog.pt
"""

import argparse
import torch
import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import dateutil.parser

import sys
sys.path.append('.')
from train_csclog import Model, getDateTimeFromISO8601String


def evaluate_model(model_path, data_dir, window_size=9, batch_size=128, device='cuda'):
    """Evaluate model by classifying sequences directly"""
    
    print("="*80)
    print("CSCLog Model Evaluation (Simple)")
    print("="*80)
    
    # Load test data
    test_normal_path = f"{data_dir}/test_normal.csv"
    test_anomaly_path = f"{data_dir}/test_anomaly.csv"
    temp_path = f"{data_dir}/log_templates.csv"
    emb_path = f"{data_dir}/sentences_emb.json"
    com_path = f"{data_dir}/component.json"
    
    # Combine test sets
    test_normal_df = pd.read_csv(test_normal_path)
    test_anomaly_df = pd.read_csv(test_anomaly_path) if os.path.exists(test_anomaly_path) else pd.DataFrame()
    test_df = pd.concat([test_normal_df, test_anomaly_df], ignore_index=True)
    
    print(f"\nTest data: {len(test_df)} sequences")
    print(f"  Normal: {len(test_df[test_df['Label']==0])}")
    print(f"  Anomaly: {len(test_df[test_df['Label']==1])}")
    
    # Load metadata
    logTemp = pd.read_csv(temp_path, index_col='EventId')
    mapping = {index: i for i, index in enumerate(logTemp.index.unique())}
    emb = json.load(open(emb_path, 'r'))
    cop = json.load(open(com_path, 'r'))
    
    attr_num = len(list(emb.items())[0][1])
    class_num = len(logTemp.index.unique())
    com_num = len(cop)
    num_keys = class_num
    
    # Load model
    print("\nLoading model...")
    hidden_size = [64, 64, 64, 64, 64]
    model = Model(attr_num, com_num, hidden_size, 0.8, 1, 2, class_num, 0.1, True).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded!")
    
    # Process sequences
    print("\nProcessing sequences...")
    all_predictions = []
    all_labels = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        try:
            seqs = eval(row['EventSequence'])
            label = row['Label']
            
            # Skip if sequence too short
            if len(seqs) < window_size:
                continue
            
            # Use first window_size events
            events = seqs[:window_size]
            
            # Encode sequence
            quan_pattern = [0] * num_keys
            log_counter = Counter([mapping[event] for event, _, _ in events])
            for key in log_counter:
                quan_pattern[key] = log_counter[key]
            
            inp, com, tm = [], [], []
            start_time = getDateTimeFromISO8601String(events[0][2])
            for event, component, time in events:
                cur_time = getDateTimeFromISO8601String(time)
                inp.append(emb[event])
                com.append(cop[str(component)] if str(component) in cop else cop.get(component, -1))
                tm.append((cur_time - start_time).seconds)
            
            # Convert to tensors
            seq_tensor = torch.tensor([inp], dtype=torch.float).to(device)
            com_tensor = torch.tensor([com]).to(device)
            quan_tensor = torch.tensor([quan_pattern], dtype=torch.float).to(device)
            time_tensor = torch.tensor([tm], dtype=torch.float).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(seq_tensor, com_tensor, quan_tensor, time_tensor)
                _, predicted = torch.max(output, 1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label)
            
        except Exception as e:
            if idx < 10:
                print(f"Error at row {idx}: {e}")
            continue
    
    print(f"\nProcessed {len(all_predictions)} sequences")
    
    if len(all_predictions) == 0:
        print("❌ No sequences processed!")
        return
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Debug: Check unique values
    print(f"\nUnique labels: {np.unique(all_labels)}")
    print(f"Unique predictions: {np.unique(all_predictions)}")
    print(f"Prediction distribution: {np.bincount(all_predictions)}")
    
    # Model predicts log template IDs, not anomaly labels!
    # Need to convert: if prediction matches actual next event = normal, else = anomaly
    # For now, use a simple heuristic: high prediction values = anomaly
    
    # Convert multiclass predictions to binary anomaly detection
    # Strategy: Use prediction confidence/entropy as anomaly score
    print("\n⚠️  Note: Model predicts log template IDs, not anomaly labels directly.")
    print("Using simple heuristic: treating all sequences as normal for now.")
    print("Proper evaluation requires comparing predicted vs actual next events.")
    
    # For demonstration, just show label distribution
    print("\n" + "="*80)
    print("LABEL DISTRIBUTION")
    print("="*80)
    print(f"Normal sequences (Label=0): {np.sum(all_labels == 0)}")
    print(f"Anomaly sequences (Label=1): {np.sum(all_labels == 1)}")
    print(f"\nModel predicted {len(np.unique(all_predictions))} different template IDs")
    print("="*80)
    
    return
    
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
    parser = argparse.ArgumentParser(description='Evaluate CSCLog model (simple)')
    parser.add_argument('--model_path', type=str, default='model/CSCLog/CSCLog.pt')
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--window_size', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_model(args.model_path, args.data_dir, args.window_size, args.batch_size, device)


if __name__ == '__main__':
    main()
