# CSCLog Documentation Index

## 📚 Quick Navigation

### 🚀 Start Here

1. **[QUICK_START.md](QUICK_START.md)** ⭐
   - One-command execution
   - 3-step deployment
   - Quick reference

2. **[COMPLETE_SOLUTION.md](COMPLETE_SOLUTION.md)** ⭐
   - Full solution overview
   - Preprocessing + Training
   - Usage examples
   - Performance metrics

### 📖 Detailed Guides

3. **[PREPROCESSING_README.md](PREPROCESSING_README.md)**
   - Preprocessing pipeline details
   - Configuration options
   - Performance tuning
   - Troubleshooting

4. **[TRAINING_README.md](TRAINING_README.md)**
   - Training script usage
   - Hyperparameters
   - Model loading
   - Advanced usage

5. **[SERVER_DEPLOYMENT_GUIDE.md](SERVER_DEPLOYMENT_GUIDE.md)**
   - Step-by-step deployment
   - Server setup
   - Monitoring
   - Troubleshooting

### 📋 Reference

6. **[README.md](README.md)**
   - Original CSCLog paper
   - Datasets
   - Citation

7. **[UPLOAD_CHECKLIST.txt](UPLOAD_CHECKLIST.txt)**
   - Pre-upload checklist
   - Files to upload
   - Verification steps

## 🎯 Use Cases

### "I want to run everything quickly"
→ [QUICK_START.md](QUICK_START.md)

### "I want to understand the complete solution"
→ [COMPLETE_SOLUTION.md](COMPLETE_SOLUTION.md)

### "I want to customize preprocessing"
→ [PREPROCESSING_README.md](PREPROCESSING_README.md)

### "I want to tune training parameters"
→ [TRAINING_README.md](TRAINING_README.md)

### "I want to deploy to server"
→ [SERVER_DEPLOYMENT_GUIDE.md](SERVER_DEPLOYMENT_GUIDE.md)

## 📂 File Structure

```
CSCLog/
├── QUICK_START.md                 ⭐ Start here
├── COMPLETE_SOLUTION.md           ⭐ Full overview
├── PREPROCESSING_README.md        📊 Preprocessing details
├── TRAINING_README.md             🚀 Training details
├── SERVER_DEPLOYMENT_GUIDE.md     🖥️ Deployment guide
├── UPLOAD_CHECKLIST.txt           ✅ Upload checklist
└── README.md                      📄 Original README
```

## 🔧 Scripts

### Main Scripts
- `run_full_pipeline.py` ⭐ - Run preprocessing + training
- `train_csclog.py` - Training only
- `run_preprocessing.py` - Preprocessing only

### Utility Scripts
- `validate_setup.py` - Validate before upload
- `test_server_setup.py` - Test server setup
- `package_for_server.py` - Create deployment package
- `setup_server.sh` - Server setup automation

## 💡 Quick Commands

### Full Pipeline
```bash
python run_full_pipeline.py
```

### Preprocessing Only
```bash
python run_preprocessing.py
```

### Training Only
```bash
python train_csclog.py
```

### Validation
```bash
python validate_setup.py
```

### Server Testing
```bash
python test_server_setup.py
```

## 📊 Expected Timeline

| Task | Time | Doc |
|------|------|-----|
| Setup | 5-10 min | SERVER_DEPLOYMENT_GUIDE.md |
| Preprocessing | 3-5 min | PREPROCESSING_README.md |
| Training | 5-10 min | TRAINING_README.md |
| **Total** | **15-25 min** | COMPLETE_SOLUTION.md |

## 🆘 Troubleshooting

### GPU Issues
→ [PREPROCESSING_README.md](PREPROCESSING_README.md#troubleshooting)
→ [TRAINING_README.md](TRAINING_README.md#troubleshooting)

### Deployment Issues
→ [SERVER_DEPLOYMENT_GUIDE.md](SERVER_DEPLOYMENT_GUIDE.md#troubleshooting)

### Data Issues
→ [PREPROCESSING_README.md](PREPROCESSING_README.md#troubleshooting)

## 📞 Support Flow

1. Check [QUICK_START.md](QUICK_START.md) for quick fixes
2. Review [COMPLETE_SOLUTION.md](COMPLETE_SOLUTION.md) for overview
3. Check specific guide for detailed troubleshooting
4. Review [UPLOAD_CHECKLIST.txt](UPLOAD_CHECKLIST.txt) for missing steps

---

**Recommended Reading Order:**
1. QUICK_START.md (5 min)
2. COMPLETE_SOLUTION.md (10 min)
3. Specific guides as needed

**For Server Deployment:**
1. UPLOAD_CHECKLIST.txt
2. SERVER_DEPLOYMENT_GUIDE.md
3. QUICK_START.md for commands
