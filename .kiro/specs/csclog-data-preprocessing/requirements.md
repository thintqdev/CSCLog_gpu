# Requirements Document

## Introduction

This document specifies the requirements for a GPU-accelerated data preprocessing pipeline that transforms raw log data (JSONL format) into the structured format required by CSCLog (Component Subsequence Correlation-Aware Log Anomaly Detection). The pipeline must handle log parsing, sequence generation, embedding creation, and component mapping while maximizing GPU utilization for computationally intensive operations.

## Glossary

- **CSCLog**: Component Subsequence Correlation-Aware Log Anomaly Detection Method, a deep learning model for detecting anomalies in system logs
- **Drain Algorithm**: A fixed-depth tree-based log parsing algorithm that extracts log templates from raw log messages
- **EventId**: A unique identifier assigned to each log template extracted by the Drain parser
- **EventSequence**: An ordered list of (EventId, Component, Timestamp) tuples representing a sequence of log events
- **Log Template**: A generalized pattern extracted from log messages by replacing variable parts with wildcards
- **Sentence Embedding**: A dense vector representation of a log template generated using pre-trained language models (BERT)
- **Component**: A system component identifier extracted from log metadata (e.g., service name, host name)
- **Sliding Window**: A fixed-size window that moves through a log sequence to create training samples
- **Session**: A logical grouping of log events based on identifiers like block_id or time windows
- **Raw Log Data**: Unprocessed log messages in JSONL format containing fields like timestamp, service, message, level, etc.
- **GPU Acceleration**: Using CUDA-enabled GPUs to parallelize computationally intensive operations like embedding generation
- **Batch Processing**: Processing multiple items simultaneously to maximize GPU utilization

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to parse raw JSONL log files using the Drain algorithm, so that I can extract structured log templates and EventIds from unstructured log messages.

#### Acceptance Criteria

1. WHEN the Preprocessing_Pipeline receives a JSONL file path, THE Preprocessing_Pipeline SHALL load all log entries into memory
2. WHEN the Drain_Parser processes log messages, THE Drain_Parser SHALL extract log templates by identifying static and dynamic parts
3. WHEN the Drain_Parser completes parsing, THE Drain_Parser SHALL assign unique EventIds to each discovered log template
4. WHEN the Drain_Parser finishes processing, THE Preprocessing_Pipeline SHALL generate a log_templates.csv file containing EventId and template mappings
5. WHEN the Drain_Parser encounters duplicate templates, THE Drain_Parser SHALL reuse existing EventIds rather than creating new ones

### Requirement 2

**User Story:** As a data scientist, I want to generate sentence embeddings for log templates using GPU-accelerated BERT models, so that I can create semantic representations efficiently for large datasets.

#### Acceptance Criteria

1. WHEN the Embedding_Generator receives log templates, THE Embedding_Generator SHALL load a pre-trained BERT model onto the GPU
2. WHEN the Embedding_Generator processes templates, THE Embedding_Generator SHALL batch templates into groups of at least 128 items to maximize V100 GPU utilization
3. WHEN the Embedding_Generator creates embeddings, THE Embedding_Generator SHALL generate 768-dimensional vectors for each unique EventId
4. WHEN the Embedding_Generator completes processing, THE Preprocessing_Pipeline SHALL save embeddings to a sentences_emb.json file with EventId as keys
5. WHILE GPU memory is available, THE Embedding_Generator SHALL increase batch size up to 256 items to improve V100 throughput

### Requirement 3

**User Story:** As a data scientist, I want to extract component information from log metadata, so that I can track which system components generated each log event.

#### Acceptance Criteria

1. WHEN the Component_Extractor processes log entries, THE Component_Extractor SHALL extract component identifiers from the service field
2. WHEN the Component_Extractor encounters a new component, THE Component_Extractor SHALL assign a unique integer component_id starting from 0
3. WHEN the Component_Extractor completes processing, THE Preprocessing_Pipeline SHALL generate a component.json file mapping component names to component_ids
4. WHEN a log entry contains multiple component fields, THE Component_Extractor SHALL prioritize the service field over the host field
5. WHEN the Component_Extractor encounters missing component information, THE Component_Extractor SHALL assign a default component_id of -1

### Requirement 4

**User Story:** As a data scientist, I want to create EventSequences from parsed logs with sliding windows, so that I can generate training samples for the CSCLog model.

#### Acceptance Criteria

1. WHEN the Sequence_Generator receives parsed logs, THE Sequence_Generator SHALL create tuples of (EventId, Component, Timestamp) for each log entry
2. WHEN the Sequence_Generator applies sliding windows, THE Sequence_Generator SHALL use a configurable window size between 6 and 14 events
3. WHEN the Sequence_Generator creates sequences, THE Sequence_Generator SHALL preserve temporal ordering based on ISO8601 timestamps
4. WHEN the Sequence_Generator processes time-based sessions, THE Sequence_Generator SHALL group logs within 10-second windows for BGL-style datasets
5. WHEN the Sequence_Generator completes processing, THE Preprocessing_Pipeline SHALL generate CSV files with EventSequence columns containing lists of tuples

### Requirement 5

**User Story:** As a data scientist, I want to split processed data into training, validation, and test sets with proper labeling, so that I can train and evaluate the CSCLog model effectively.

#### Acceptance Criteria

1. WHEN the Data_Splitter receives processed sequences, THE Data_Splitter SHALL allocate 70 percent of normal sequences to training data
2. WHEN the Data_Splitter creates validation sets, THE Data_Splitter SHALL allocate 15 percent of normal sequences and 50 percent of anomaly sequences to validation
3. WHEN the Data_Splitter creates test sets, THE Data_Splitter SHALL allocate 15 percent of normal sequences and 50 percent of anomaly sequences to testing
4. WHEN the Data_Splitter generates output files, THE Preprocessing_Pipeline SHALL create train_normal.csv, test_normal.csv, and test_anomaly.csv files
5. WHEN sequences lack anomaly labels, THE Data_Splitter SHALL treat all sequences as normal by default

### Requirement 6

**User Story:** As a data scientist, I want to monitor GPU utilization and processing progress in real-time, so that I can optimize performance and identify bottlenecks.

#### Acceptance Criteria

1. WHEN the Preprocessing_Pipeline starts processing, THE Progress_Monitor SHALL display a progress bar showing percentage completion
2. WHEN the Embedding_Generator uses GPU, THE GPU_Monitor SHALL report current GPU memory usage in megabytes
3. WHEN each processing stage completes, THE Progress_Monitor SHALL log the elapsed time in seconds
4. WHEN GPU memory exceeds 90 percent capacity, THE Embedding_Generator SHALL reduce batch size by 50 percent
5. WHILE processing continues, THE Progress_Monitor SHALL update statistics every 5 seconds

### Requirement 7

**User Story:** As a data scientist, I want to configure pipeline parameters through a configuration file, so that I can easily adjust settings without modifying code.

#### Acceptance Criteria

1. WHEN the Preprocessing_Pipeline initializes, THE Configuration_Loader SHALL read parameters from a config.yaml file
2. WHEN the Configuration_Loader reads settings, THE Configuration_Loader SHALL validate that window_size is between 6 and 14
3. WHEN the Configuration_Loader reads settings, THE Configuration_Loader SHALL validate that batch_size is a positive integer less than 256
4. WHEN configuration parameters are missing, THE Configuration_Loader SHALL use default values of window_size equals 9 and batch_size equals 64
5. WHEN the Configuration_Loader detects invalid parameters, THE Preprocessing_Pipeline SHALL raise a validation error with descriptive messages

### Requirement 8

**User Story:** As a data scientist, I want the pipeline to handle large datasets efficiently through streaming and chunking, so that I can process datasets larger than available RAM.

#### Acceptance Criteria

1. WHEN the Preprocessing_Pipeline processes files larger than 1 gigabyte, THE File_Loader SHALL use streaming to read data in chunks of 500 megabytes
2. WHEN the Embedding_Generator processes large template sets, THE Embedding_Generator SHALL process embeddings in batches and save incrementally
3. WHEN the Sequence_Generator creates sequences, THE Sequence_Generator SHALL write output to disk every 50000 sequences to prevent memory overflow
4. WHEN available RAM drops below 20 percent, THE Preprocessing_Pipeline SHALL trigger garbage collection
5. WHILE processing 4 million records, THE Memory_Monitor SHALL track peak memory usage and log warnings when usage exceeds 80 percent
