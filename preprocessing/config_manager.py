"""
Configuration Manager for CSCLog Preprocessing Pipeline
Loads and validates configuration from YAML files
"""

import os
import yaml
from typing import Any, Dict
from pathlib import Path


class ConfigurationError(Exception):
    """Configuration validation errors"""
    pass


class ConfigManager:
    """Manages pipeline configuration with validation"""
    
    DEFAULT_CONFIG = {
        'input': {
            'raw_log_path': 'dataset/data_full.jsonl',
            'output_dir': 'dataset/processed'
        },
        'parsing': {
            'depth': 4,
            'similarity_threshold': 0.4,
            'max_children': 100,
            'chunk_size': 500000
        },
        'embedding': {
            'model_name': 'model/bert',
            'batch_size': 128,
            'max_batch_size': 256,
            'device': 'cuda',
            'max_length': 512,
            'use_fp16': True,
            'use_tf32': True,
            'pin_memory': True
        },
        'sequence': {
            'window_size': 9,
            'session_type': 'sliding',
            'time_window_seconds': 10,
            'write_interval': 50000
        },
        'splitting': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42
        },
        'monitoring': {
            'enable_gpu_monitor': True,
            'enable_progress_bar': True,
            'log_interval_seconds': 5,
            'memory_warning_threshold': 0.8
        },
        'performance': {
            'num_workers': 4,
            'prefetch_factor': 2,
            'chunk_size_mb': 500,
            'gc_interval': 100000
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._merge_config(user_config)
            print(f"Loaded configuration from {config_path}")
        else:
            if config_path:
                print(f"Warning: Config file {config_path} not found. Using defaults.")
            else:
                print("Using default configuration")
        
        self.validate()
    
    def _merge_config(self, user_config: Dict):
        """Recursively merge user config with defaults"""
        def merge_dict(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, user_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation
        
        Args:
            key: Configuration key (e.g., 'embedding.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def validate(self) -> bool:
        """
        Validate all configuration parameters
        
        Returns:
            True if valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate window_size
        window_size = self.window_size
        if not (6 <= window_size <= 14):
            raise ConfigurationError(
                f"window_size must be between 6 and 14, got {window_size}"
            )
        
        # Validate batch_size
        batch_size = self.batch_size
        if not (1 <= batch_size <= 256):
            raise ConfigurationError(
                f"batch_size must be between 1 and 256, got {batch_size}"
            )
        
        # Validate max_batch_size
        max_batch_size = self.max_batch_size
        if max_batch_size < batch_size:
            raise ConfigurationError(
                f"max_batch_size ({max_batch_size}) must be >= batch_size ({batch_size})"
            )
        
        # Validate split ratios
        train_ratio = self.get('splitting.train_ratio')
        val_ratio = self.get('splitting.val_ratio')
        test_ratio = self.get('splitting.test_ratio')
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ConfigurationError(
                f"Split ratios must sum to 1.0, got {total_ratio}"
            )
        
        # Validate file paths
        raw_log_path = self.get('input.raw_log_path')
        if not os.path.exists(raw_log_path):
            raise ConfigurationError(
                f"Input log file not found: {raw_log_path}"
            )
        
        # Create output directory if it doesn't exist
        output_dir = self.get('input.output_dir')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Configuration validation passed")
        return True
    
    @property
    def window_size(self) -> int:
        """Get sliding window size (6-14)"""
        return self.get('sequence.window_size', 9)
    
    @property
    def batch_size(self) -> int:
        """Get batch size for GPU processing (1-256)"""
        return self.get('embedding.batch_size', 128)
    
    @property
    def max_batch_size(self) -> int:
        """Get maximum batch size"""
        return self.get('embedding.max_batch_size', 256)
    
    @property
    def bert_model(self) -> str:
        """Get BERT model path or name"""
        return self.get('embedding.model_name', 'model/bert')
    
    @property
    def device(self) -> str:
        """Get device (cuda/cpu)"""
        return self.get('embedding.device', 'cuda')
    
    @property
    def use_fp16(self) -> bool:
        """Check if FP16 is enabled"""
        return self.get('embedding.use_fp16', True)
    
    def __repr__(self) -> str:
        return f"ConfigManager(window_size={self.window_size}, batch_size={self.batch_size}, device={self.device})"
