# CSCLog Preprocessing - Server Deployment Guide

## Chuẩn bị trước khi upload

### 1. Validate code trên máy local (không chạy preprocessing)

```bash
python validate_setup.py
```

Kiểm tra xem tất cả files cần thiết đã có chưa.

### 2. Tạo package để upload

```bash
python package_for_server.py
```

Sẽ tạo file `csclog_preprocessing.zip` (~200-500MB tùy BERT model).

## Upload lên Server

### Option 1: Upload qua SCP

```bash
# Upload package
scp csclog_preprocessing.zip user@server:/path/to/destination/

# Upload data file riêng (nếu lớn)
scp dataset/data_full.jsonl user@server:/path/to/destination/dataset/
```

### Option 2: Upload qua SFTP

```bash
sftp user@server
put csclog_preprocessing.zip
put dataset/data_full.jsonl
```

### Option 3: Clone từ Git (nếu có repo)

```bash
ssh user@server
git clone <your-repo-url>
cd <repo-name>
```

## Setup trên Server

### 1. SSH vào server

```bash
ssh user@server
```

### 2. Extract package (nếu dùng zip)

```bash
unzip csclog_preprocessing.zip
cd csclog_preprocessing
```

### 3. Kiểm tra GPU

```bash
nvidia-smi
```

Kết quả mong đợi:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.xx.xx    Driver Version: 470.xx.xx    CUDA Version: 11.4   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-FHHL...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P0    25W / 250W |      0MiB / 16384MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 4. Chạy setup script

```bash
chmod +x setup_server.sh
./setup_server.sh
```

Script này sẽ:
- Tạo virtual environment
- Install dependencies
- Kiểm tra CUDA/GPU
- Validate setup
- Chạy test

### 5. Kiểm tra config

```bash
cat config.yaml
```

Đảm bảo các settings đúng:
```yaml
input:
  raw_log_path: "dataset/data_full.jsonl"  # ✓ Đúng path
  output_dir: "dataset/processed"

embedding:
  device: "cuda"        # ✓ Dùng GPU
  batch_size: 128       # ✓ Phù hợp V100 16GB
  use_fp16: true        # ✓ Bật Tensor Cores
```

### 6. Test setup

```bash
source venv/bin/activate  # Activate virtual environment
python test_server_setup.py
```

Kết quả mong đợi: Tất cả tests PASS ✓

## Chạy Preprocessing

### 1. Dry run (kiểm tra config)

```bash
python run_preprocessing.py --dry-run
```

### 2. Chạy full preprocessing

```bash
# Chạy trong tmux/screen để không bị disconnect
tmux new -s preprocessing

# Hoặc screen
screen -S preprocessing

# Chạy preprocessing
python run_preprocessing.py

# Monitor GPU trong terminal khác
watch -n 1 nvidia-smi
```

### 3. Detach khỏi session (nếu dùng tmux/screen)

```bash
# Tmux: Ctrl+B, then D
# Screen: Ctrl+A, then D
```

### 4. Reattach để xem progress

```bash
# Tmux
tmux attach -t preprocessing

# Screen
screen -r preprocessing
```

## Monitoring

### Xem GPU usage

```bash
watch -n 1 nvidia-smi
```

### Xem logs real-time

```bash
tail -f preprocessing.log  # Nếu có log file
```

### Kiểm tra progress

Pipeline sẽ hiển thị:
```
[Stage 1/5] Log Parsing with Drain Algorithm
--------------------------------------------------------------------------------
Loading logs from dataset/data_full.jsonl...
Loading logs: 100%|████████████████████| 4000000/4000000 [00:45<00:00, 88888.89it/s]
Parsing 4000000 log messages with Drain algorithm...
Processed 100.0% of log lines.
✓ Parsed 4,000,000 logs into 1,234 templates
  Time: 80.23s

[Stage 2/5] GPU-Accelerated Embedding Generation
--------------------------------------------------------------------------------
Using device: cuda
Loading BERT model from model/bert...
Model loaded successfully on cuda
Model converted to FP16 for Tensor Core acceleration
Generating embeddings: 100%|████████████| 1234/1234 [00:35<00:00, 35.11it/s]
✓ Generated 1,234 embeddings
  Time: 35.12s
  GPU Memory: 8234.5MB / 16384.0MB (50.3%)
...
```

## Kết quả

### Kiểm tra outputs

```bash
ls -lh dataset/processed/
```

Kết quả mong đợi:
```
-rw-r--r-- 1 user group  1.2M log_templates.csv
-rw-r--r-- 1 user group   45M sentences_emb.json
-rw-r--r-- 1 user group  1.5K component.json
-rw-r--r-- 1 user group  850M train_normal.csv
-rw-r--r-- 1 user group  180M test_normal.csv
-rw-r--r-- 1 user group   15M test_anomaly.csv
-rw-r--r-- 1 user group  2.3K preprocessing_report.json
```

### Xem report

```bash
cat dataset/processed/preprocessing_report.json
```

### Verify với CSCLog training

```python
# Trong main.ipynb
name = 'dataset/processed'
train_path = name + '/train_normal.csv'
test_normal_path = name + '/test_normal.csv'
test_anomaly_path = name + '/test_anomaly.csv'
temp_path = name + '/log_templates.csv'
emb_path = name + '/sentences_emb.json'
com_path = name + '/component.json'

# Load và check
import pandas as pd
train_df = pd.read_csv(train_path)
print(f"Training samples: {len(train_df)}")
```

## Troubleshooting

### GPU Out of Memory

```bash
# Giảm batch size trong config.yaml
embedding:
  batch_size: 64  # Giảm từ 128
  max_batch_size: 128  # Giảm từ 256
```

### CUDA not available

```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, check driver
nvidia-smi

# Reinstall PyTorch với CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Slow performance

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Nếu GPU util thấp, tăng batch size
embedding:
  batch_size: 256  # Tăng lên
```

### Process killed

```bash
# Check memory
free -h

# Nếu RAM không đủ, giảm chunk size
parsing:
  chunk_size: 250000  # Giảm từ 500000
```

## Expected Timeline (4M records, V100 16GB)

| Stage | Time | Progress |
|-------|------|----------|
| Parsing | ~80s | 40% |
| Embedding | ~60s | 70% |
| Component | ~10s | 75% |
| Sequence | ~40s | 95% |
| Splitting | ~10s | 100% |
| **Total** | **~3-5 min** | |

## Cleanup

### Xóa intermediate files

```bash
rm -rf dataset/processed/temp_*
```

### Deactivate environment

```bash
deactivate
```

### Exit tmux/screen

```bash
# Tmux
tmux kill-session -t preprocessing

# Screen
screen -X -S preprocessing quit
```

## Next Steps

Sau khi preprocessing xong:

1. ✓ Verify outputs trong `dataset/processed/`
2. ✓ Check `preprocessing_report.json`
3. ✓ Update paths trong `main.ipynb`
4. ✓ Run CSCLog training

## Support

Nếu gặp vấn đề:
1. Check `preprocessing_report.json`
2. Run `python test_server_setup.py`
3. Check GPU: `nvidia-smi`
4. Review logs với `--verbose`
