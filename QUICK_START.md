# CSCLog Complete Pipeline - Quick Start

## 🚀 Nhanh nhất (1 lệnh)

```bash
# Chạy cả preprocessing + training
python run_full_pipeline.py
```

## 📋 Chi tiết (3 bước)

### Trên máy local:

```bash
# 1. Validate
python validate_setup.py

# 2. Package
python package_for_server.py
```

### Trên server:

```bash
# 3. Upload, setup và chạy FULL PIPELINE
scp csclog_preprocessing.zip user@server:/path/
ssh user@server
unzip csclog_preprocessing.zip && cd csclog_preprocessing
./setup_server.sh

# Chạy cả preprocessing + training
python run_full_pipeline.py
```

## 🎯 Hoặc chạy từng bước

```bash
# Bước 1: Preprocessing
python run_preprocessing.py

# Bước 2: Training
python train_csclog.py
```

## 📋 Checklist nhanh

### Trước khi upload:
- [ ] `python validate_setup.py` - PASS
- [ ] `python package_for_server.py` - Tạo zip
- [ ] Upload `csclog_preprocessing.zip` lên server
- [ ] Upload `dataset/data_full.jsonl` (nếu chưa có)

### Trên server:
- [ ] Extract zip
- [ ] `./setup_server.sh` - Setup environment
- [ ] `python test_server_setup.py` - Test GPU
- [ ] `python run_full_pipeline.py` - Chạy FULL PIPELINE

### Kết quả:
- [ ] `dataset/processed/` có 7 files (preprocessing)
- [ ] `model/CSCLog/CSCLog.pt` có model (training)
- [ ] Review metrics: Accuracy, Precision, Recall, F1

## ⚡ Expected Performance

**Tesla V100 16GB với 4M records:**

### Preprocessing
- Total time: ~3-5 phút
- Throughput: ~15,000-20,000 logs/giây
- GPU memory: ~8-12GB
- Output size: ~1.1GB

### Training (2 epochs, 50K sequences)
- Total time: ~5-10 phút
- GPU memory: ~8-12GB
- Model size: ~50MB
- Best F1: ~0.90+

## 📁 Files cần upload

```
csclog_preprocessing/
├── preprocessing/          # Core code
├── utils/Drain.py         # Log parser
├── model/bert/            # BERT model
├── config.yaml            # Configuration
├── run_preprocessing.py   # Main script
└── requirements_preprocessing.txt
```

Plus: `dataset/data_full.jsonl` (upload riêng nếu lớn)

## 🎯 Outputs

```
dataset/processed/
├── log_templates.csv      # 1,234 templates
├── sentences_emb.json     # 768-dim embeddings
├── component.json         # Component mapping
├── train_normal.csv       # 70% normal data
├── test_normal.csv        # 15% normal data
├── test_anomaly.csv       # Anomaly data (if any)
└── preprocessing_report.json
```

## 🔧 Common Issues

**GPU OOM?**
```yaml
embedding:
  batch_size: 64  # Giảm từ 128
```

**Slow?**
```bash
nvidia-smi  # Check GPU util
# Nếu thấp, tăng batch_size
```

**CUDA not found?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 📚 Docs

- `PREPROCESSING_README.md` - Full documentation
- `SERVER_DEPLOYMENT_GUIDE.md` - Detailed deployment guide
- `config.yaml` - Configuration reference

## 💡 Tips

1. **Dùng tmux/screen** để không bị disconnect:
   ```bash
   tmux new -s preprocessing
   python run_preprocessing.py
   # Ctrl+B, D để detach
   ```

2. **Monitor GPU** trong terminal khác:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Check progress** real-time:
   - Pipeline hiển thị progress bars
   - Stage timing sau mỗi stage
   - GPU memory usage

## ✅ Success Criteria

Preprocessing thành công khi:
- ✓ Tất cả 5 stages complete
- ✓ 7 output files được tạo
- ✓ `preprocessing_report.json` có đầy đủ stats
- ✓ CSCLog training load được data

## 🎓 Next: Training

```python
# main.ipynb
name = 'dataset/processed'
train_path = name + '/train_normal.csv'
# ... continue with CSCLog training
```

---

**Questions?** Check `PREPROCESSING_README.md` hoặc `SERVER_DEPLOYMENT_GUIDE.md`
