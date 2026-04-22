"""
Test script for server setup
Run this on server to verify everything works before full preprocessing
"""

import sys
import os

def test_gpu():
    """Test GPU availability and specs"""
    print("\n" + "="*80)
    print("Testing GPU Setup")
    print("="*80)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                
                # Check if it's V100
                if "V100" in props.name:
                    print("  ✓ Tesla V100 detected - Optimal for this pipeline")
                    print("  ✓ Tensor Cores available for FP16")
            
            # Test tensor creation
            print("\nTesting GPU tensor operations...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("  ✓ GPU tensor operations working")
            
            # Test FP16
            print("\nTesting FP16 support...")
            x_fp16 = x.half()
            y_fp16 = y.half()
            z_fp16 = torch.matmul(x_fp16, y_fp16)
            print("  ✓ FP16 operations working")
            
            return True
        else:
            print("⚠ No GPU available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_bert_model():
    """Test BERT model loading"""
    print("\n" + "="*80)
    print("Testing BERT Model")
    print("="*80)
    
    try:
        from transformers import BertTokenizer, BertModel
        import torch
        
        model_path = "model/bert"
        
        if not os.path.exists(model_path):
            print(f"⚠ BERT model not found at {model_path}")
            print("  Will need to download or provide model files")
            return False
        
        print(f"Loading BERT model from {model_path}...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        
        print("  ✓ Model loaded successfully")
        
        # Test inference
        print("\nTesting BERT inference...")
        text = "This is a test log message"
        inputs = tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        print(f"  ✓ Generated embedding: shape {embedding.shape}")
        print(f"  ✓ Embedding dimension: {embedding.shape[1]} (expected: 768)")
        
        return True
        
    except Exception as e:
        print(f"❌ BERT model test failed: {e}")
        return False

def test_preprocessing_imports():
    """Test preprocessing module imports"""
    print("\n" + "="*80)
    print("Testing Preprocessing Imports")
    print("="*80)
    
    try:
        from preprocessing import ConfigManager, PreprocessingPipeline
        print("  ✓ ConfigManager imported")
        print("  ✓ PreprocessingPipeline imported")
        
        from preprocessing.drain_parser import DrainParser
        print("  ✓ DrainParser imported")
        
        from preprocessing.embedding_generator import EmbeddingGenerator
        print("  ✓ EmbeddingGenerator imported")
        
        from preprocessing.component_extractor import ComponentExtractor
        print("  ✓ ComponentExtractor imported")
        
        from preprocessing.sequence_generator import SequenceGenerator
        print("  ✓ SequenceGenerator imported")
        
        from preprocessing.data_splitter import DataSplitter
        print("  ✓ DataSplitter imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test configuration loading"""
    print("\n" + "="*80)
    print("Testing Configuration")
    print("="*80)
    
    try:
        from preprocessing import ConfigManager
        
        config = ConfigManager('config.yaml')
        print(f"  ✓ Config loaded: {config}")
        print(f"\nConfiguration values:")
        print(f"  Input path: {config.get('input.raw_log_path')}")
        print(f"  Output dir: {config.get('input.output_dir')}")
        print(f"  Window size: {config.window_size}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Device: {config.device}")
        print(f"  Use FP16: {config.use_fp16}")
        print(f"  BERT model: {config.bert_model}")
        
        # Validate
        config.validate()
        print("\n  ✓ Configuration is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_small_sample():
    """Test preprocessing on small sample"""
    print("\n" + "="*80)
    print("Testing Small Sample Processing")
    print("="*80)
    
    try:
        # Create small test file
        test_data = """app.kolla: [1776673441.920507907, {"service":"gnocchi","source":"kolla","timestamp":"2026-04-20T17:24:01.918+09:00","level":"ERROR","filepath":"/var/log/kolla/gnocchi/gnocchi-metricd.log","message":"Unhandled exception","pid":"2418226","host":"controller"}]
app.kolla: [1776673441.922440052, {"service":"gnocchi","source":"kolla","timestamp":"2026-04-20T17:24:01.920+09:00","level":"ERROR","filepath":"/var/log/kolla/gnocchi/gnocchi-metricd.log","message":"Unhandled exception","pid":"2418225","host":"controller"}]
app.kolla: [1776673441.931360960, {"service":"cinder","source":"kolla","timestamp":"2026-04-20T17:24:01.929+09:00","level":"INFO","filepath":"/var/log/kolla/cinder/cinder-api.log","message":"Service started successfully","pid":"2418228","host":"controller"}]"""
        
        test_file = "test_sample.jsonl"
        with open(test_file, 'w') as f:
            f.write(test_data)
        
        print(f"Created test file: {test_file}")
        
        # Test parsing
        from preprocessing.drain_parser import DrainParser
        parser = DrainParser()
        
        print("\nTesting log parsing...")
        templates_df, event_mapping = parser.parse(test_file, "test_output")
        print(f"  ✓ Parsed {len(event_mapping)} logs")
        print(f"  ✓ Found {len(templates_df)} templates")
        
        # Cleanup
        os.remove(test_file)
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        
        print("\n  ✓ Small sample test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("CSCLog Preprocessing - Server Setup Test")
    print("="*80)
    
    tests = [
        ("GPU Setup", test_gpu),
        ("Preprocessing Imports", test_preprocessing_imports),
        ("Configuration", test_config),
        ("BERT Model", test_bert_model),
        ("Small Sample", test_small_sample),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Server is ready for preprocessing.")
        print("\nYou can now run:")
        print("  python run_preprocessing.py --dry-run")
        print("  python run_preprocessing.py")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix issues before running preprocessing.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
