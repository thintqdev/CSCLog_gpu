"""Quick test for ConfigManager"""

from preprocessing.config_manager import ConfigManager

# Test with config file
config = ConfigManager("config.yaml")
print(f"\n{config}")
print(f"Window size: {config.window_size}")
print(f"Batch size: {config.batch_size}")
print(f"Device: {config.device}")
print(f"Use FP16: {config.use_fp16}")
print(f"BERT model: {config.bert_model}")
print(f"Output dir: {config.get('input.output_dir')}")
print("\nConfiguration loaded successfully!")
