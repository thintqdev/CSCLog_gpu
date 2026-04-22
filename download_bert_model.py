#!/usr/bin/env python
"""
Download BERT model for CSCLog preprocessing
Run this on server before preprocessing
"""

from transformers import BertTokenizer, BertModel
import os

print("="*80)
print("Downloading BERT Model for CSCLog")
print("="*80)

# Model to download
model_name = "bert-base-uncased"
save_path = "model/bert"

print(f"\nDownloading {model_name}...")
print(f"Save to: {save_path}")
print("\nThis may take a few minutes...")

# Create directory
os.makedirs(save_path, exist_ok=True)

# Download tokenizer
print("\n[1/2] Downloading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
print("✓ Tokenizer downloaded")

# Download model
print("\n[2/2] Downloading model...")
model = BertModel.from_pretrained(model_name)
model.save_pretrained(save_path)
print("✓ Model downloaded")

# Verify
print("\n" + "="*80)
print("Verification")
print("="*80)

files = os.listdir(save_path)
print(f"\nFiles in {save_path}:")
for f in sorted(files):
    size = os.path.getsize(os.path.join(save_path, f)) / (1024*1024)
    print(f"  ✓ {f} ({size:.1f} MB)")

print("\n" + "="*80)
print("✅ BERT model downloaded successfully!")
print("="*80)
print("\nYou can now run preprocessing:")
print("  python run_full_pipeline.py")
