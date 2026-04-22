# CSCLog Data Preprocessing Pipeline

GPU-accelerated preprocessing pipeline for transforming raw JSONL log files into structured datasets for CSCLog anomaly detection training.

## Features

- 🚀 **GPU-Accelerated**: Optimized for Tesla V100 16GB with FP16 mixed precision
- 📊 **Drain Algorithm**: Efficient log parsing and template extraction
- 🤖 **BERT Embeddings**: Semantic log template representations
- 🔄 **Adaptive Batching**: Dynamic batch size adjustment based on GPU memory
- 💾 **Memory Efficient**: Streaming processing for large datasets (4M+ records)
- ⚡ **High Throughput**: ~15,000-20,000 logs/second end-to-end

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM (for 4M records)
- Tesla V100 16GB or similar GPU

### Install Dependencies

```bash
pip install -r requirements_preprocessing.txt
```

### Verify Installation

```bash
python test_config.py
```

## Quick Start

### 1. Prepare Configuration

Edit `config.yaml` to match your setup:

```yaml
input:
  raw_log_path: "dataset/data_full.jsonl"
  output_dir: "dataset/processed"

embedding:
  model_name: "model/bert"  # Path to local BERT model
  batch_size: 128           # Optimized for V100
  device: "cuda"
  use_fp16: true            # Enable Tensor Cores
```

### 2. Run Pipeline

```bash
python run_preprocessing.py
```

### 3. Check Outputs

The pipeline generates the following files in `dataset/processed/`:

- `log_templates.csv` - Extracted log templates with EventIds
- `sentences_emb.json` - BERT embeddings for each template
- `component.json` - Component name to ID mapping
- `train_normal.csv` - Training data (normal sequences only)
- `test_normal.csv` - Test data (normal sequences)
- `test_anomaly.csv` - Test data (anomaly sequences)
- `preprocessing_report.json` - Processing statistics

## Usage

### Basic Usage

```bash
# Run with default config.yaml
python run_preprocessing.py

# Run with custom config
python run_preprocessing.py --config my_config.yaml

# Validate configuration without running
python run_preprocessing.py --dry-run

# Verbose output
python run_preprocessing.py --verbose
```

### Configuration Options

#### Input/Output

```yaml
input:
  raw_log_path: "dataset/data_full.jsonl"  # Input JSONL file
  output_dir: "dataset/processed"          # Output directory
```

#### Parsing (Drain Algorithm)

```yaml
parsing:
  depth: 4                      # Tree depth (3-5 recommended)
  similarity_threshold: 0.4     # Template matching threshold (0.3-0.5)
  max_children: 100             # Max children per node
  chunk_size: 500000            # Process N logs at a time
```

#### Embedding Generation

```yaml
embedding:
  model_name: "model/bert"      # BERT model path
  batch_size: 128               # Initial batch size (V100: 128-256)
  max_batch_size: 256           # Maximum batch size
  device: "cuda"                # "cuda" or "cpu"
  max_length: 512               # Max sequence length
  use_fp16: true                # Enable FP16 (V100 Tensor Cores)
  use_tf32: true                # Enable TF32 (V100 only)
  pin_memory: true              # Faster CPU-GPU transfers
```

#### Sequence Generation

```yaml
sequence:
  window_size: 9                # Sliding window size (6-14)
  session_type: "sliding"       # "sliding" or "time_window"
  time_window_seconds: 10       # For time_window mode
  write_interval: 50000         # Write every N sequences
```

#### Data Splitting

```yaml
splitting:
  train_ratio: 0.7              # 70% normal data for training
  val_ratio: 0.15               # 15% for validation
  test_ratio: 0.15              # 15% for testing
  random_seed: 42               # For reproducibility
```

## Performance Tuning

### GPU Optimization

**For Tesla V100 16GB:**
- `batch_size: 128` - Good starting point
- `max_batch_size: 256` - Allow adaptive scaling
- `use_fp16: true` - 2-3x speedup with Tensor Cores
- `use_tf32: true` - Additional speedup

**For smaller GPUs (8GB):**
- `batch_size: 64`
- `max_batch_size: 128`
- `use_fp16: true`

**For CPU only:**
- `device: "cpu"`
- `batch_size: 32`
- `use_fp16: false`

### Memory Management

**For large datasets (4M+ records):**
- `chunk_size: 500000` - Process 500K logs at a time
- `write_interval: 50000` - Write every 50K sequences
- `chunk_size_mb: 500` - Read 500MB chunks

**For smaller datasets (<1M records):**
- `chunk_size: 100000`
- `write_interval: 10000`
- `chunk_size_mb: 100`

## Expected Performance

### Tesla V100 16GB with 4M Records

- **Parsing**: ~50,000 logs/second
- **Embedding**: ~2,000-3,000 templates/second (with FP16)
- **End-to-end**: ~15,000-20,000 logs/second
- **Total time**: ~3-5 minutes for 4M records

### Breakdown by Stage

| Stage | Time | Percentage |
|-------|------|------------|
| Parsing | ~80s | 40% |
| Embedding | ~60s | 30% |
| Component | ~10s | 5% |
| Sequence | ~40s | 20% |
| Splitting | ~10s | 5% |

## Troubleshooting

### GPU Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` in config.yaml
2. Enable `use_fp16: true` for lower memory usage
3. Close other GPU applications
4. Use smaller `max_batch_size`

### Slow Performance

**Check**:
1. GPU utilization: `nvidia-smi`
2. Batch size too small: Increase `batch_size`
3. CPU bottleneck: Increase `num_workers`
4. Disk I/O: Use SSD for input/output

### Invalid Configuration

**Symptoms**: `ConfigurationError`

**Solutions**:
1. Run `python run_preprocessing.py --dry-run`
2. Check `window_size` is between 6-14
3. Check `batch_size` is between 1-256
4. Verify input file exists

### Malformed JSONL

**Symptoms**: Warnings about skipping lines

**Solutions**:
- Check JSONL format: `app.kolla: [timestamp, {data}]`
- Validate JSON with: `python -m json.tool < data.jsonl`
- Pipeline will skip invalid lines automatically

## Output Format

### log_templates.csv

```csv
EventId,EventTemplate,Occurrences
E001,"Unhandled exception",7
E002,"Manager for service <*> <*> is reporting problems",2
```

### sentences_emb.json

```json
{
  "E001": [0.123, -0.456, 0.789, ...],  // 768 dimensions
  "E002": [0.234, -0.567, 0.890, ...]
}
```

### component.json

```json
{
  "gnocchi": 0,
  "cinder": 1,
  "fluent-bit": 2
}
```

### train_normal.csv

```csv
SessionId,EventSequence,Label
S000001,"[('E001', 0, '2026-04-20T17:24:01.918+09:00'), ...]",0
```

## Integration with CSCLog Training

After preprocessing, use the generated files with CSCLog:

```python
# In main.ipynb
name = 'dataset/processed'
train_path = name + '/train_normal.csv'
test_normal_path = name + '/test_normal.csv'
test_anomaly_path = name + '/test_anomaly.csv'
temp_path = name + '/log_templates.csv'
emb_path = name + '/sentences_emb.json'
com_path = name + '/component.json'

# Continue with CSCLog training...
```

## Advanced Usage

### Python API

```python
from preprocessing import ConfigManager, PreprocessingPipeline

# Load config
config = ConfigManager("config.yaml")

# Run pipeline
pipeline = PreprocessingPipeline(config)
pipeline.run()

# Check statistics
print(f"Processed {pipeline.stats['num_logs']} logs")
print(f"Generated {pipeline.stats['num_sequences']} sequences")
```

### Custom Processing

```python
from preprocessing import DrainParser, EmbeddingGenerator

# Parse logs only
parser = DrainParser(depth=4, similarity_threshold=0.4)
templates_df, event_mapping = parser.parse("logs.jsonl", "output/")

# Generate embeddings only
generator = EmbeddingGenerator(model_name="bert-base-uncased")
embeddings = generator.generate_embeddings(templates)
```

## FAQ

**Q: Can I use a different BERT model?**  
A: Yes, set `model_name` to any HuggingFace model name or local path.

**Q: How do I process multiple log files?**  
A: Concatenate JSONL files first: `cat file1.jsonl file2.jsonl > combined.jsonl`

**Q: Can I run without GPU?**  
A: Yes, set `device: "cpu"` in config. Processing will be slower (~10x).

**Q: How do I label anomalies?**  
A: Currently all sequences are labeled as normal (0). Manual labeling required for anomaly detection.

**Q: What if I have different log format?**  
A: Modify `DrainParser._load_jsonl()` to parse your format.

## Support

For issues or questions:
1. Check this README
2. Review `preprocessing_report.json` for statistics
3. Run with `--verbose` for detailed logs
4. Check GPU with `nvidia-smi`

## License

MIT License - See main CSCLog repository for details.
