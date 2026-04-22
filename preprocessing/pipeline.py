"""
Main Preprocessing Pipeline for CSCLog
Orchestrates the entire preprocessing workflow
"""

import os
import time
import json
from pathlib import Path
from typing import Dict

from .config_manager import ConfigManager
from .drain_parser import DrainParser
from .embedding_generator import EmbeddingGenerator
from .component_extractor import ComponentExtractor
from .sequence_generator import SequenceGenerator
from .data_splitter import DataSplitter


class PreprocessingPipeline:
    """Main pipeline orchestrator for CSCLog preprocessing"""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize pipeline with configuration
        
        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.stats = {
            'total_time': 0,
            'parsing_time': 0,
            'embedding_time': 0,
            'component_time': 0,
            'sequence_time': 0,
            'splitting_time': 0,
            'num_logs': 0,
            'num_templates': 0,
            'num_components': 0,
            'num_sequences': 0
        }
    
    def run(self):
        """Execute the complete preprocessing pipeline"""
        print("="*80)
        print("CSCLog Data Preprocessing Pipeline")
        print("="*80)
        print(f"Configuration: {self.config}")
        print(f"Input: {self.config.get('input.raw_log_path')}")
        print(f"Output: {self.config.get('input.output_dir')}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Stage 1: Log Parsing
            print("\n[Stage 1/5] Log Parsing with Drain Algorithm")
            print("-"*80)
            templates_df, event_mapping, parsed_logs = self._stage_parsing()
            
            # Stage 2: Embedding Generation
            print("\n[Stage 2/5] GPU-Accelerated Embedding Generation")
            print("-"*80)
            embeddings = self._stage_embedding(templates_df)
            
            # Stage 3: Component Extraction
            print("\n[Stage 3/5] Component Extraction")
            print("-"*80)
            component_ids, component_map = self._stage_component_extraction(parsed_logs)
            
            # Stage 4: Sequence Generation
            print("\n[Stage 4/5] EventSequence Generation")
            print("-"*80)
            sequences_df = self._stage_sequence_generation(event_mapping, component_ids)
            
            # Stage 5: Data Splitting
            print("\n[Stage 5/5] Data Splitting")
            print("-"*80)
            self._stage_data_splitting(sequences_df)
            
            # Calculate total time
            self.stats['total_time'] = time.time() - start_time
            
            # Generate summary report
            self._generate_report()
            
            print("\n" + "="*80)
            print("Pipeline completed successfully!")
            print("="*80)
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _stage_parsing(self):
        """Stage 1: Parse logs with Drain algorithm"""
        start_time = time.time()
        
        parser = DrainParser(
            depth=self.config.get('parsing.depth', 4),
            similarity_threshold=self.config.get('parsing.similarity_threshold', 0.4),
            max_children=self.config.get('parsing.max_children', 100),
            chunk_size=self.config.get('parsing.chunk_size', 500000)
        )
        
        input_path = self.config.get('input.raw_log_path')
        output_dir = self.config.get('input.output_dir')
        
        templates_df, event_mapping = parser.parse(input_path, output_dir)
        
        # Save templates
        templates_path = os.path.join(output_dir, "log_templates.csv")
        parser.save_templates(templates_path)
        
        # Load parsed logs for component extraction
        parsed_logs = parser._load_jsonl(input_path, progress_bar=False)
        
        self.stats['parsing_time'] = time.time() - start_time
        self.stats['num_logs'] = len(event_mapping)
        self.stats['num_templates'] = len(templates_df)
        
        print(f"✓ Parsed {self.stats['num_logs']} logs into {self.stats['num_templates']} templates")
        print(f"  Time: {self.stats['parsing_time']:.2f}s")
        
        return templates_df, event_mapping, parsed_logs
    
    def _stage_embedding(self, templates_df):
        """Stage 2: Generate embeddings with GPU"""
        start_time = time.time()
        
        generator = EmbeddingGenerator(
            model_name=self.config.bert_model,
            batch_size=self.config.batch_size,
            max_batch_size=self.config.max_batch_size,
            device=self.config.device,
            max_length=self.config.get('embedding.max_length', 512),
            use_fp16=self.config.use_fp16,
            use_tf32=self.config.get('embedding.use_tf32', True),
            pin_memory=self.config.get('embedding.pin_memory', True)
        )
        
        # Create template dict: EventId -> template_text
        templates = {}
        for _, row in templates_df.iterrows():
            templates[row['EventId']] = row['EventTemplate']
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(templates)
        
        # Save embeddings
        output_dir = self.config.get('input.output_dir')
        embeddings_path = os.path.join(output_dir, "sentences_emb.json")
        generator.save_embeddings(embeddings, embeddings_path)
        
        self.stats['embedding_time'] = time.time() - start_time
        
        # Get GPU stats if available
        if self.config.device == "cuda":
            gpu_stats = generator.get_gpu_memory_stats()
            print(f"✓ Generated {len(embeddings)} embeddings")
            print(f"  Time: {self.stats['embedding_time']:.2f}s")
            print(f"  GPU Memory: {gpu_stats['used_mb']:.1f}MB / {gpu_stats['total_mb']:.1f}MB ({gpu_stats['percentage']:.1%})")
        else:
            print(f"✓ Generated {len(embeddings)} embeddings")
            print(f"  Time: {self.stats['embedding_time']:.2f}s")
        
        return embeddings
    
    def _stage_component_extraction(self, parsed_logs):
        """Stage 3: Extract component information"""
        start_time = time.time()
        
        extractor = ComponentExtractor()
        component_ids, component_map = extractor.extract_components(parsed_logs)
        
        # Save component mapping
        output_dir = self.config.get('input.output_dir')
        component_path = os.path.join(output_dir, "component.json")
        extractor.save_mapping(component_path)
        
        self.stats['component_time'] = time.time() - start_time
        self.stats['num_components'] = len(component_map)
        
        print(f"✓ Extracted {self.stats['num_components']} unique components")
        print(f"  Time: {self.stats['component_time']:.2f}s")
        
        return component_ids, component_map
    
    def _stage_sequence_generation(self, event_mapping, component_ids):
        """Stage 4: Generate EventSequences"""
        start_time = time.time()
        
        generator = SequenceGenerator(
            window_size=self.config.window_size,
            session_type=self.config.get('sequence.session_type', 'sliding'),
            time_window_seconds=self.config.get('sequence.time_window_seconds', 10),
            write_interval=self.config.get('sequence.write_interval', 50000)
        )
        
        input_path = self.config.get('input.raw_log_path')
        output_dir = self.config.get('input.output_dir')
        sequences_path = os.path.join(output_dir, "sequences.csv")
        
        sequences_df = generator.generate_sequences(
            input_path, event_mapping, component_ids, sequences_path
        )
        
        self.stats['sequence_time'] = time.time() - start_time
        self.stats['num_sequences'] = len(sequences_df)
        
        print(f"✓ Generated {self.stats['num_sequences']} sequences")
        print(f"  Time: {self.stats['sequence_time']:.2f}s")
        
        return sequences_df
    
    def _stage_data_splitting(self, sequences_df):
        """Stage 5: Split data into train/val/test"""
        start_time = time.time()
        
        splitter = DataSplitter(
            train_ratio=self.config.get('splitting.train_ratio', 0.7),
            val_ratio=self.config.get('splitting.val_ratio', 0.15),
            test_ratio=self.config.get('splitting.test_ratio', 0.15),
            random_seed=self.config.get('splitting.random_seed', 42)
        )
        
        train_df, val_df, test_df = splitter.split(sequences_df)
        
        output_dir = self.config.get('input.output_dir')
        splitter.save_splits(train_df, val_df, test_df, output_dir)
        
        self.stats['splitting_time'] = time.time() - start_time
        
        print(f"✓ Split data into train/val/test sets")
        print(f"  Time: {self.stats['splitting_time']:.2f}s")
    
    def _generate_report(self):
        """Generate processing summary report"""
        output_dir = self.config.get('input.output_dir')
        report_path = os.path.join(output_dir, "preprocessing_report.json")
        
        report = {
            'config': {
                'window_size': self.config.window_size,
                'batch_size': self.config.batch_size,
                'device': self.config.device,
                'use_fp16': self.config.use_fp16
            },
            'statistics': self.stats,
            'output_files': {
                'templates': 'log_templates.csv',
                'embeddings': 'sentences_emb.json',
                'components': 'component.json',
                'sequences': 'sequences.csv',
                'train': 'train_normal.csv',
                'val_normal': 'val_normal.csv',
                'val_anomaly': 'val_anomaly.csv',
                'test_normal': 'test_normal.csv',
                'test_anomaly': 'test_anomaly.csv'
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Processing Summary")
        print(f"{'='*80}")
        print(f"Total logs processed:     {self.stats['num_logs']:,}")
        print(f"Unique templates:         {self.stats['num_templates']:,}")
        print(f"Unique components:        {self.stats['num_components']:,}")
        print(f"Generated sequences:      {self.stats['num_sequences']:,}")
        print(f"\nTiming Breakdown:")
        print(f"  Parsing:                {self.stats['parsing_time']:.2f}s")
        print(f"  Embedding generation:   {self.stats['embedding_time']:.2f}s")
        print(f"  Component extraction:   {self.stats['component_time']:.2f}s")
        print(f"  Sequence generation:    {self.stats['sequence_time']:.2f}s")
        print(f"  Data splitting:         {self.stats['splitting_time']:.2f}s")
        print(f"  Total time:             {self.stats['total_time']:.2f}s")
        print(f"\nThroughput: {self.stats['num_logs'] / self.stats['total_time']:.0f} logs/second")
        print(f"\nReport saved to: {report_path}")
