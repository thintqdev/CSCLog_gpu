# Implementation Plan

- [x] 1. Set up project structure and configuration system


  - Create directory structure for preprocessing pipeline
  - Implement ConfigManager class to load and validate YAML configuration
  - Create default config.yaml template with all parameters
  - Add configuration validation for window_size, batch_size, and file paths
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_



- [ ] 2. Implement Drain log parser
  - [ ] 2.1 Create DrainParser class with tree-based parsing logic
    - Implement Drain tree structure with configurable depth
    - Add log tokenization and preprocessing methods
    - Implement template matching with similarity threshold

    - Add EventId assignment and template storage
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [ ] 2.2 Add log loading and JSONL parsing
    - Implement streaming JSONL file reader for large files

    - Extract message field from log entries
    - Handle malformed JSON entries gracefully
    - _Requirements: 1.1, 8.1_
  
  - [ ] 2.3 Implement template output generation
    - Create log_templates.csv with EventId, EventTemplate, Occurrences columns


    - Generate event_mapping dictionary for downstream processing
    - Add template statistics logging
    - _Requirements: 1.4_

- [x] 3. Implement GPU-accelerated embedding generator

  - [ ] 3.1 Create EmbeddingGenerator class with BERT model loading
    - Load pre-trained BERT model from local path
    - Initialize model on GPU with proper device management
    - Implement CPU fallback when GPU unavailable
    - Add model warmup to optimize first batch
    - _Requirements: 2.1_
  

  - [ ] 3.2 Implement adaptive batch processing for embeddings
    - Create dynamic batching logic with initial batch size of 128 (optimized for V100)
    - Implement GPU memory monitoring during processing
    - Add automatic batch size adjustment between 64-256 based on memory usage
    - Implement OOM error handling with batch size reduction
    - Clear CUDA cache between batches
    - _Requirements: 2.2, 2.5, 6.4_
  
  - [ ] 3.3 Add embedding generation and storage
    - Generate 768-dimensional embeddings for each EventId
    - Implement batch encoding with torch.no_grad() for efficiency
    - Save embeddings to sentences_emb.json with EventId keys
    - Add progress tracking for embedding generation


    - _Requirements: 2.3, 2.4_
  
  - [ ]* 3.4 Optimize GPU performance for V100
    - Implement mixed precision (FP16) inference using V100 Tensor Cores
    - Add memory pinning for faster CPU-GPU transfers
    - Enable TF32 mode for additional V100 speedup
    - _Requirements: 2.2, 2.5_

- [ ] 4. Implement component extraction
  - [ ] 4.1 Create ComponentExtractor class
    - Initialize component mapping dictionary
    - Implement component name extraction with priority logic (service > host)
    - Assign unique integer component_ids starting from 0
    - Handle missing component fields with default value -1
    - _Requirements: 3.1, 3.2, 3.4, 3.5_
  
  - [x] 4.2 Generate component mapping output

    - Create component.json file with name-to-id mappings
    - Add component statistics logging
    - _Requirements: 3.3_

- [ ] 5. Implement sequence generator
  - [x] 5.1 Create SequenceGenerator class with sliding window logic


    - Implement sliding window algorithm with configurable window size
    - Create EventSequence tuples with (EventId, ComponentId, Timestamp) format
    - Preserve temporal ordering based on ISO8601 timestamps
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 5.2 Add time-based session grouping

    - Implement 10-second time window grouping for BGL-style datasets
    - Handle timestamp parsing with dateutil
    - Group logs within time windows
    - _Requirements: 4.4_
  
  - [x] 5.3 Generate sequence output files

    - Create CSV files with SessionId, EventSequence, Label columns
    - Implement incremental writing for 4M records (every 50000 sequences)
    - Format EventSequence as list of tuples
    - _Requirements: 4.5, 8.3_

- [ ] 6. Implement data splitter
  - [x] 6.1 Create DataSplitter class with stratified splitting



    - Implement train/val/test split with configurable ratios
    - Apply stratified sampling to maintain label distribution
    - Use random seed for reproducibility
    - _Requirements: 5.1, 5.2, 5.3_
  
  - [x] 6.2 Generate split output files

    - Create train_normal.csv with 70% of normal sequences
    - Create test_normal.csv and test_anomaly.csv with proper splits
    - Handle datasets without anomaly labels (all normal)
    - _Requirements: 5.4, 5.5_

- [ ] 7. Implement monitoring system
  - [ ] 7.1 Create GPUMonitor class
    - Implement GPU memory usage tracking with torch.cuda
    - Add GPU utilization percentage monitoring
    - Report memory stats in megabytes
    - _Requirements: 6.2_
  
  - [ ] 7.2 Create ProgressMonitor class with tqdm
    - Implement progress bar with percentage completion
    - Add elapsed time tracking for each stage
    - Update statistics every 5 seconds
    - Display current processing rate
    - _Requirements: 6.1, 6.3, 6.5_
  
  - [ ] 7.3 Integrate monitoring into pipeline stages
    - Add progress tracking to log parsing
    - Add GPU monitoring to embedding generation
    - Add progress tracking to sequence generation
    - Log stage completion times
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 8. Implement memory management for 4M records
  - [ ] 8.1 Add streaming file processing
    - Implement chunked JSONL reading for large files (500MB chunks)
    - Process 4M records in manageable batches
    - Add memory usage monitoring
    - _Requirements: 8.1, 8.5_
  
  - [ ] 8.2 Add memory optimization utilities
    - Implement automatic garbage collection when RAM <20%
    - Add peak memory usage tracking
    - Log warnings when memory usage >80%
    - _Requirements: 8.4, 8.5_

- [ ] 9. Create main pipeline orchestrator
  - [x] 9.1 Implement PreprocessingPipeline class


    - Initialize all components with configuration
    - Orchestrate execution flow: parse → embed → extract → sequence → split
    - Handle errors and cleanup resources
    - Generate processing summary report
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1_
  
  - [x] 9.2 Add command-line interface


    - Create CLI with argparse for config file path
    - Add options for individual stage execution
    - Implement verbose logging mode
    - Add dry-run mode for validation
    - _Requirements: 7.1_
  
  - [x] 9.3 Implement error handling and recovery

    - Add try-catch blocks for each pipeline stage
    - Implement checkpoint saving for resume capability
    - Add cleanup utilities for partial outputs
    - Log all errors with stack traces
    - _Requirements: All error handling requirements_

- [ ] 10. Create example and documentation
  - [x] 10.1 Create example configuration file

    - Provide config.yaml template with comments
    - Document all configuration parameters
    - Include examples for different use cases
    - _Requirements: 7.1_
  
  - [x] 10.2 Write usage documentation


    - Create README with installation instructions
    - Document pipeline execution steps
    - Add troubleshooting guide
    - Include performance tuning tips
    - _Requirements: All requirements_
  
  - [x] 10.3 Create example dataset

    - Generate small test dataset (100 logs)
    - Create expected output files
    - Add validation script
    - _Requirements: All requirements_

- [ ]* 11. Write tests
  - [ ]* 11.1 Write unit tests for core components
    - Test ConfigManager validation logic
    - Test DrainParser template extraction
    - Test ComponentExtractor mapping logic
    - Test SequenceGenerator window logic
    - Test DataSplitter split ratios
    - _Requirements: All requirements_
  
  - [ ]* 11.2 Write integration tests
    - Test end-to-end pipeline with small dataset
    - Test GPU vs CPU execution paths
    - Test error handling and recovery
    - _Requirements: All requirements_
  
  - [ ]* 11.3 Write performance tests
    - Benchmark throughput with different dataset sizes
    - Measure GPU utilization and memory usage
    - Profile memory usage and identify bottlenecks
    - _Requirements: 6.2, 8.5_

- [x] 12. Optimize and validate

  - [ ] 12.1 Run pipeline on sample data
    - Execute pipeline on dataset/data_full.jsonl
    - Verify all output files are generated correctly
    - Validate output format matches CSCLog requirements
    - _Requirements: All requirements_


  
  - [ ] 12.2 Validate outputs with CSCLog training
    - Load generated files into CSCLog training script
    - Verify data shapes and formats are correct
    - Test training loop with processed data
    - _Requirements: All requirements_
  
  - [ ]* 12.3 Performance tuning
    - Optimize batch sizes for target GPU
    - Tune memory management parameters
    - Optimize file I/O operations
    - _Requirements: 2.5, 6.4, 8.1_
