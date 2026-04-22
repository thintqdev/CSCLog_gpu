"""
Package preprocessing pipeline for server deployment
Creates a zip file with all necessary files
"""

import os
import zipfile
from pathlib import Path
import shutil

def create_deployment_package():
    """Create deployment package for server"""
    
    print("Creating deployment package for server...")
    
    # Files and directories to include
    include_items = [
        # Core preprocessing code
        'preprocessing/',
        'utils/Drain.py',
        'utils/__init__.py',
        
        # Configuration and scripts
        'config.yaml',
        'run_preprocessing.py',
        'requirements_preprocessing.txt',
        'PREPROCESSING_README.md',
        
        # Model (if exists locally)
        'model/bert/',
        
        # Data (optional - might be too large)
        # 'dataset/data_full.jsonl',
    ]
    
    # Create temporary directory
    temp_dir = 'deployment_package'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"\nCopying files to {temp_dir}/...")
    
    # Copy files
    for item in include_items:
        if os.path.exists(item):
            if os.path.isdir(item):
                dest = os.path.join(temp_dir, item)
                shutil.copytree(item, dest)
                print(f"  ✓ Copied directory: {item}")
            else:
                dest = os.path.join(temp_dir, item)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(item, dest)
                print(f"  ✓ Copied file: {item}")
        else:
            print(f"  ⚠ Skipped (not found): {item}")
    
    # Create deployment instructions
    instructions = """
# CSCLog Preprocessing Pipeline - Server Deployment

## Quick Start

1. Extract this package on server:
   ```bash
   unzip csclog_preprocessing.zip
   cd csclog_preprocessing
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\\Scripts\\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements_preprocessing.txt
   ```

4. Upload your data:
   - Place data_full.jsonl in dataset/ directory
   - Or update config.yaml with correct path

5. Verify setup:
   ```bash
   python run_preprocessing.py --dry-run
   ```

6. Run preprocessing:
   ```bash
   python run_preprocessing.py
   ```

7. Monitor GPU (if using CUDA):
   ```bash
   watch -n 1 nvidia-smi
   ```

## Configuration

Edit config.yaml before running:
- Set correct paths for input/output
- Adjust batch_size for your GPU
- Enable/disable FP16 based on GPU support

## Outputs

Results will be in dataset/processed/:
- log_templates.csv
- sentences_emb.json
- component.json
- train_normal.csv
- test_normal.csv
- test_anomaly.csv
- preprocessing_report.json

## Troubleshooting

See PREPROCESSING_README.md for detailed documentation.

For GPU issues:
- Check CUDA: python -c "import torch; print(torch.cuda.is_available())"
- Check GPU: nvidia-smi
- Reduce batch_size if OOM errors

## Support

Check preprocessing_report.json for statistics and timing information.
    """
    
    with open(os.path.join(temp_dir, 'DEPLOYMENT_INSTRUCTIONS.md'), 'w') as f:
        f.write(instructions)
    print("  ✓ Created DEPLOYMENT_INSTRUCTIONS.md")
    
    # Create zip file
    zip_name = 'csclog_preprocessing.zip'
    print(f"\nCreating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
                
    # Get zip size
    zip_size = os.path.getsize(zip_name) / (1024 * 1024)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ Deployment package created: {zip_name} ({zip_size:.1f} MB)")
    print("\nNext steps:")
    print("1. Upload csclog_preprocessing.zip to your server")
    print("2. Upload dataset/data_full.jsonl separately (if not included)")
    print("3. Follow instructions in DEPLOYMENT_INSTRUCTIONS.md")
    
    return zip_name

if __name__ == '__main__':
    create_deployment_package()
