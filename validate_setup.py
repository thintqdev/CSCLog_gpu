"""
Validation script to check if preprocessing pipeline is ready for server deployment
Run this before uploading to server
"""

import sys
import os

def check_files():
    """Check if all required files exist"""
    print("Checking required files...")
    
    required_files = [
        'config.yaml',
        'run_preprocessing.py',
        'requirements_preprocessing.txt',
        'preprocessing/__init__.py',
        'preprocessing/config_manager.py',
        'preprocessing/pipeline.py',
        'preprocessing/drain_parser.py',
        'preprocessing/embedding_generator.py',
        'preprocessing/component_extractor.py',
        'preprocessing/sequence_generator.py',
        'preprocessing/data_splitter.py',
        'utils/Drain.py',
        'dataset/data_full.jsonl',
        'model/bert/config.json',
        'model/bert/vocab.txt',
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} required files!")
        return False
    
    print("\n✓ All required files present")
    return True

def check_imports():
    """Check if all imports work"""
    print("\nChecking Python imports...")
    
    try:
        from preprocessing import ConfigManager, PreprocessingPipeline
        print("  ✓ preprocessing module")
    except ImportError as e:
        print(f"  ✗ preprocessing module - {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        print("  ✗ torch - NOT INSTALLED")
        return False
    
    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers - NOT INSTALLED")
        return False
    
    try:
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
    except ImportError:
        print("  ✗ pandas - NOT INSTALLED")
        return False
    
    try:
        import yaml
        print("  ✓ pyyaml")
    except ImportError:
        print("  ✗ pyyaml - NOT INSTALLED")
        return False
    
    print("\n✓ All imports successful")
    return True

def check_config():
    """Check if config is valid"""
    print("\nValidating configuration...")
    
    try:
        from preprocessing import ConfigManager
        config = ConfigManager('config.yaml')
        print(f"  ✓ Config loaded: {config}")
        print(f"    - Input: {config.get('input.raw_log_path')}")
        print(f"    - Output: {config.get('input.output_dir')}")
        print(f"    - Device: {config.device}")
        print(f"    - Batch size: {config.batch_size}")
        print(f"    - Window size: {config.window_size}")
        return True
    except Exception as e:
        print(f"  ✗ Config validation failed: {e}")
        return False

def check_data():
    """Check if input data exists and is valid"""
    print("\nChecking input data...")
    
    try:
        import json
        data_path = 'dataset/data_full.jsonl'
        
        if not os.path.exists(data_path):
            print(f"  ✗ Data file not found: {data_path}")
            return False
        
        # Check file size
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  ✓ Data file exists: {size_mb:.1f} MB")
        
        # Check first few lines
        with open(data_path, 'r', encoding='utf-8') as f:
            line_count = 0
            valid_count = 0
            for i, line in enumerate(f):
                line_count += 1
                if i >= 10:  # Check first 10 lines
                    break
                try:
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            data = json.loads(parts[1].strip())
                            if isinstance(data, list) and len(data) >= 2:
                                valid_count += 1
                except:
                    pass
        
        print(f"  ✓ Validated {valid_count}/10 sample lines")
        
        if valid_count < 5:
            print("  ⚠ Warning: Many invalid lines detected")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data validation failed: {e}")
        return False

def generate_deployment_checklist():
    """Generate checklist for server deployment"""
    print("\n" + "="*80)
    print("DEPLOYMENT CHECKLIST FOR SERVER")
    print("="*80)
    
    checklist = """
1. Files to upload to server:
   □ preprocessing/ directory (all .py files)
   □ utils/ directory (Drain.py and other utilities)
   □ model/bert/ directory (BERT model files)
   □ dataset/data_full.jsonl
   □ config.yaml
   □ run_preprocessing.py
   □ requirements_preprocessing.txt

2. Server setup commands:
   □ Create virtual environment: python -m venv venv
   □ Activate environment: source venv/bin/activate
   □ Install dependencies: pip install -r requirements_preprocessing.txt
   □ Verify CUDA: python -c "import torch; print(torch.cuda.is_available())"
   □ Check GPU: nvidia-smi

3. Configuration for server:
   □ Update config.yaml paths if needed
   □ Set device: "cuda" for GPU
   □ Set batch_size: 128 for V100 16GB
   □ Enable use_fp16: true for Tensor Cores

4. Run preprocessing:
   □ Dry run first: python run_preprocessing.py --dry-run
   □ Full run: python run_preprocessing.py
   □ Monitor with: watch -n 1 nvidia-smi

5. Verify outputs in dataset/processed/:
   □ log_templates.csv
   □ sentences_emb.json
   □ component.json
   □ train_normal.csv
   □ test_normal.csv
   □ test_anomaly.csv (if anomalies exist)
   □ preprocessing_report.json

6. Use with CSCLog training:
   □ Update paths in main.ipynb
   □ Run training notebook
    """
    
    print(checklist)

def main():
    """Run all validation checks"""
    print("="*80)
    print("CSCLog Preprocessing Pipeline - Deployment Validation")
    print("="*80)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Data", check_data),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! Ready for server deployment.")
        generate_deployment_checklist()
        return 0
    else:
        print("\n❌ Some checks failed. Please fix issues before deployment.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
