# CSCLog Training Guide

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

Chạy cả preprocessing và training trong 1 lệnh:

```bash
python run_full_pipeline.py
```

### Option 2: Training Only

Nếu đã có preprocessed data:

```bash
python train_csclog.py
```

### Option 3: Step by Step

```bash
# 1. Preprocessing
python run_preprocessing.py

# 2. Training
python train_csclog.py
```

## 📋 Commands

### Full Pipeline

```bash
# Default (2 epochs)
python run_full_pipeline.py

# Custom epochs
python run_full_pipeline.py --epochs 10

# Skip preprocessing (if already done)
python run_full_pipeline.py --skip-preprocessing

# Only preprocessing
python run_full_pipeline.py --skip-training

# Custom parameters
python run_full_pipeline.py --epochs 5 --batch-size 32 --lr 0.0001
```

### Training Only

```bash
# Default settings
python train_csclog.py

# Custom data directory
python train_csclog.py --data_dir dataset/processed

# Custom hyperparameters
python train_csclog.py --num_epochs 10 --batch_size 32 --lr 0.0001

# Custom window size
python train_csclog.py --window_size 9

# Save to different directory
python train_csclog.py --model_dir model/my_model
```

## ⚙️ Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `dataset/processed` | Preprocessed data directory |
| `--model_dir` | `model/CSCLog` | Model save directory |
| `--num_epochs` | `2` | Number of training epochs |
| `--batch_size` | `16` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--window_size` | `9` | Sliding window size |
| `--seed` | `42` | Random seed |

### Model Hyperparameters (in code)

```python
hidden_size = [64, 64, 64, 64, 64]  # ft, lstm, mlp, gcn, out
alpha = 0.8                          # Feature encoder ratio
pattern = 1                          # Encoder pattern
num_layers = 2                       # LSTM layers
drop = 0.1                           # Dropout rate
```

## 📊 Expected Output

### Training Progress

```
================================================================================
CSCLog Training
================================================================================
Data directory: dataset/processed
Window size: 9
Batch size: 16
Epochs: 2
Learning rate: 0.001
================================================================================

Loading training data...
Processing training data: 100%|████████████| 50000/50000 [00:10<00:00]
Encoding sequences: 100%|████████████████████| 45000/45000 [00:15<00:00]
Number of train_seqs: 45000, components: 5

Loading validation data...
Processing test_normal: 100%|████████████████| 5000/5000 [00:02<00:00]
Number of test_normal_seqs(session): 5000
Processing test_anomaly: 100%|██████████████| 1000/1000 [00:00<00:00]
Number of test_anomaly_seqs(session): 1000

Creating model...
Variable inited.
x:64, sen_x:51, time_x:13
1,234,567 total parameters.

================================================================================
Starting Training
================================================================================
Epoch [1/2]: 100%|████████████████| 2813/2813 [02:15<00:00, loss: 0.3456]
Epoch [1/2], train_loss: 0.3456
TopK=1 | Accuracy: 0.923, Precision: 0.891, Recall: 0.876, F1-score: 0.883

Epoch [2/2]: 100%|████████████████| 2813/2813 [02:12<00:00, loss: 0.2134]
Epoch [2/2], train_loss: 0.2134
TopK=1 | Accuracy: 0.945, Precision: 0.912, Recall: 0.901, F1-score: 0.906

Best epoch: 2 / Best F1: 0.906

Model saved to: model/CSCLog/CSCLog.pt

================================================================================
Training Complete!
================================================================================
```

## 📁 Input Files Required

Training script expects these files in `--data_dir`:

```
dataset/processed/
├── log_templates.csv      # From preprocessing
├── sentences_emb.json     # From preprocessing
├── component.json         # From preprocessing
├── train_normal.csv       # From preprocessing
├── test_normal.csv        # From preprocessing
└── test_anomaly.csv       # From preprocessing (optional)
```

## 💾 Output

### Model File

```
model/CSCLog/CSCLog.pt
```

Contains:
- `model`: Model state dict
- `optimizer`: Optimizer state dict
- `epoch`: Best epoch number
- `f1`: Best F1 score

### Load Model

```python
import torch

# Load checkpoint
checkpoint = torch.load('model/CSCLog/CSCLog.pt')

# Load model
model.load_state_dict(checkpoint['model'])
model.eval()

# Check best epoch
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best F1: {checkpoint['f1']}")
```

## 🎯 Performance

### Expected Training Time (V100 16GB)

| Dataset Size | Epochs | Time per Epoch | Total Time |
|--------------|--------|----------------|------------|
| 50K sequences | 2 | ~2-3 min | ~5 min |
| 100K sequences | 2 | ~5-6 min | ~12 min |
| 500K sequences | 2 | ~20-25 min | ~45 min |

### GPU Memory Usage

- Training: ~8-12GB
- Batch size 16: ~8GB
- Batch size 32: ~12GB
- Batch size 64: ~15GB (may OOM on 16GB GPU)

## 🔧 Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size
python train_csclog.py --batch_size 8

# Or use CPU (slower)
# Model will auto-detect and use CPU if CUDA unavailable
```

### Data Not Found

```bash
# Check if preprocessing was done
ls -lh dataset/processed/

# If not, run preprocessing first
python run_preprocessing.py
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if GPU underutilized
python train_csclog.py --batch_size 32
```

### Window Size Mismatch

Ensure window_size matches between preprocessing and training:
- Preprocessing: `window_size: 9` in config.yaml
- Training: `--window_size 9`

## 📈 Monitoring

### Watch GPU

```bash
watch -n 1 nvidia-smi
```

### Training in Background

```bash
# Using tmux
tmux new -s training
python train_csclog.py
# Ctrl+B, D to detach

# Reattach
tmux attach -t training

# Using nohup
nohup python train_csclog.py > training.log 2>&1 &

# Check progress
tail -f training.log
```

## 🧪 Validation

### Check Model Performance

```python
import torch
from train_csclog import Model

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('model/CSCLog/CSCLog.pt')

# Create model (need same hyperparameters)
model = Model(attr_num, com_num, hidden_size, alpha, pattern,
             num_layers, class_num, drop, False).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Run inference
# ... your inference code
```

## 🎓 Advanced Usage

### Custom Hyperparameters

Edit `train_csclog.py` to modify:

```python
# Line ~650
hidden_size = [64, 64, 64, 64, 64]  # Increase for larger model
alpha = 0.8                          # Adjust feature ratio
pattern = 1                          # Try different patterns (0, 1, 2)
num_layers = 2                       # More layers = more capacity
drop = 0.1                           # Increase to prevent overfitting
```

### Early Stopping

Add early stopping by modifying training loop:

```python
patience = 5
no_improve = 0

for epoch in range(num_epochs):
    # ... training code
    
    if F1 > best_f1:
        best_f1 = F1
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print("Early stopping!")
        break
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# In training loop
scheduler.step(F1)
```

## 📊 Metrics

Training reports:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall

## 🔄 Retraining

To retrain from scratch:

```bash
# Remove old model
rm -rf model/CSCLog/CSCLog.pt

# Train again
python train_csclog.py
```

To continue training:

```python
# Load checkpoint
checkpoint = torch.load('model/CSCLog/CSCLog.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

# Continue training
# ... training loop
```

## 📝 Notes

- Training script auto-detects GPU/CPU
- Progress bars show real-time loss
- Best model is saved based on F1 score
- Validation runs after each epoch
- All random seeds are set for reproducibility

## 🆘 Support

If training fails:
1. Check GPU: `nvidia-smi`
2. Check data: `ls dataset/processed/`
3. Check logs for error messages
4. Try reducing batch size
5. Verify preprocessing completed successfully

---

**Ready to train!** 🚀
