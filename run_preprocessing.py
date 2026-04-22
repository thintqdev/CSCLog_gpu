#!/usr/bin/env python
"""
CSCLog Data Preprocessing Pipeline - Command Line Interface
Usage: python run_preprocessing.py [--config config.yaml] [--verbose]
"""

import argparse
import sys
import os
from preprocessing.config_manager import ConfigManager, ConfigurationError
from preprocessing.pipeline import PreprocessingPipeline


def main():
    """Main entry point for preprocessing pipeline"""
    parser = argparse.ArgumentParser(
        description='CSCLog Data Preprocessing Pipeline - GPU-accelerated log preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python run_preprocessing.py
  
  # Run with custom config
  python run_preprocessing.py --config my_config.yaml
  
  # Run with verbose logging
  python run_preprocessing.py --verbose
  
  # Dry run (validate config only)
  python run_preprocessing.py --dry-run
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running pipeline'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['parse', 'embed', 'component', 'sequence', 'split', 'all'],
        default='all',
        help='Run specific stage only (default: all)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = ConfigManager(args.config)
        
        if args.dry_run:
            print("\n✓ Configuration is valid!")
            print("\nConfiguration summary:")
            print(f"  Input:        {config.get('input.raw_log_path')}")
            print(f"  Output:       {config.get('input.output_dir')}")
            print(f"  Window size:  {config.window_size}")
            print(f"  Batch size:   {config.batch_size}")
            print(f"  Device:       {config.device}")
            print(f"  Use FP16:     {config.use_fp16}")
            print("\nDry run complete. Pipeline not executed.")
            return 0
        
        # Create and run pipeline
        pipeline = PreprocessingPipeline(config)
        
        if args.stage == 'all':
            pipeline.run()
        else:
            print(f"\n[INFO] Running stage: {args.stage}")
            print("[WARNING] Individual stage execution not yet implemented.")
            print("[INFO] Running full pipeline instead...")
            pipeline.run()
        
        return 0
        
    except ConfigurationError as e:
        print(f"\n[ERROR] Configuration error: {e}")
        return 1
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        return 1
    
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_banner():
    """Print ASCII banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   CSCLog Data Preprocessing Pipeline                                     ║
║   GPU-Accelerated Log Preprocessing for Anomaly Detection                ║
║                                                                           ║
║   Optimized for Tesla V100 16GB                                          ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == '__main__':
    sys.exit(main())
