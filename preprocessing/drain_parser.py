"""
Drain Log Parser Wrapper for CSCLog Preprocessing Pipeline
Adapts the existing Drain parser for JSONL input
"""

import json
import os
import pandas as pd
import hashlib
from typing import Dict, List, Tuple
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.Drain import LogParser as DrainLogParser


class DrainParser:
    """Wrapper for Drain parser to handle JSONL input"""
    
    def __init__(self, depth: int = 4, similarity_threshold: float = 0.4, 
                 max_children: int = 100, chunk_size: int = 500000):
        """
        Initialize Drain parser
        
        Args:
            depth: Drain tree depth
            similarity_threshold: Similarity threshold for template matching
            max_children: Max number of children per node
            chunk_size: Number of logs to process at once
        """
        self.depth = depth
        self.st = similarity_threshold
        self.max_children = max_children
        self.chunk_size = chunk_size
        self.templates_df = None
        self.event_mapping = {}
        
    def parse(self, jsonl_path: str, output_dir: str, progress_bar: bool = True) -> Tuple[pd.DataFrame, Dict[int, str]]:
        """
        Parse JSONL log file and extract templates
        
        Args:
            jsonl_path: Path to input JSONL file
            output_dir: Directory to save output files
            progress_bar: Show progress bar
            
        Returns:
            templates_df: DataFrame with EventId, EventTemplate, Occurrences
            event_mapping: Dict mapping log_index -> EventId
        """
        print(f"Loading logs from {jsonl_path}...")
        
        # Load JSONL and convert to format expected by Drain
        logs = self._load_jsonl(jsonl_path, progress_bar)
        
        # Create temporary CSV for Drain parser
        temp_csv = os.path.join(output_dir, "temp_logs.csv")
        os.makedirs(output_dir, exist_ok=True)
        
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(temp_csv, index=False)
        
        print(f"Parsing {len(logs)} log messages with Drain algorithm...")
        
        # Use Drain parser
        # Format: <Timestamp> <Level> <Component> <Content>
        log_format = '<Timestamp> <Level> <Component> <Content>'
        
        parser = DrainLogParser(
            log_format=log_format,
            indir=output_dir,
            outdir=output_dir,
            depth=self.depth,
            st=self.st,
            maxChild=self.max_children,
            keep_para=False,  # Disable ParameterList to avoid DataFrame error
            rex=[
                r'(\d+\.){3}\d+',  # IP addresses
                r'\d{2}:\d{2}:\d{2}',  # Time
                r'\d+',  # Numbers
            ]
        )
        
        parser.parse("temp_logs.csv")
        
        # Load results
        templates_path = os.path.join(output_dir, "temp_logs_templates.csv")
        structured_path = os.path.join(output_dir, "temp_logs_structured.csv")
        
        self.templates_df = pd.read_csv(templates_path)
        structured_df = pd.read_csv(structured_path)
        
        # Create event mapping: log_index -> EventId
        self.event_mapping = {}
        for idx, row in structured_df.iterrows():
            self.event_mapping[idx] = row['EventId']
        
        # Clean up temp files
        os.remove(temp_csv)
        
        print(f"Parsing complete. Found {len(self.templates_df)} unique templates.")
        
        return self.templates_df, self.event_mapping
    
    def _load_jsonl(self, jsonl_path: str, progress_bar: bool = True) -> List[Dict]:
        """
        Load JSONL file and extract log messages
        Supports 2 formats:
        1. app.kolla: [timestamp, {log_data}]
        2. {"message": "...", "module": "...", "@timestamp": "..."}
        
        Args:
            jsonl_path: Path to JSONL file
            progress_bar: Show progress bar
            
        Returns:
            List of log dictionaries with Timestamp, Level, Component, Content
        """
        logs = []
        
        # Count total lines for progress bar
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            iterator = tqdm(f, total=total_lines, desc="Loading logs", disable=False) if progress_bar else f
            
            for line_num, line in enumerate(iterator):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Debug first 3 lines
                    if line_num < 3:
                        tqdm.write(f"Debug line {line_num}: {line[:80]}...")
                    
                    # Try to detect format
                    # Format 1: prefix: [timestamp, {data}]
                    if ':' in line and line.startswith(('app.', 'remote.')):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            prefix = parts[0].strip()
                            json_part = parts[1].strip()
                            
                            data = json.loads(json_part)
                            if isinstance(data, list) and len(data) >= 2:
                                log_data = data[1]
                                
                                timestamp = log_data.get('timestamp', '')
                                level = log_data.get('level', 'INFO')
                                component = log_data.get('service', log_data.get('host', 'unknown'))
                                message = log_data.get('message', '')
                                
                                if line_num < 3:
                                    tqdm.write(f"  Format 1: timestamp={timestamp}, component={component}")
                                
                                if message and timestamp:
                                    logs.append({
                                        'Timestamp': timestamp,
                                        'Level': level,
                                        'Component': component,
                                        'Content': message
                                    })
                    
                    # Format 2: {"message": "...", "module": "...", "@timestamp": "..."}
                    else:
                        log_data = json.loads(line)
                        
                        # Extract fields
                        message = log_data.get('message', '')
                        timestamp = log_data.get('@timestamp', log_data.get('timestamp', ''))
                        level = log_data.get('level', log_data.get('severity', 'INFO'))
                        component = log_data.get('module', log_data.get('service', log_data.get('host', 'unknown')))
                        
                        if line_num < 3:
                            tqdm.write(f"  Format 2: timestamp={timestamp}, component={component}, message={message[:30]}")
                        
                        if message and timestamp:
                            logs.append({
                                'Timestamp': timestamp,
                                'Level': level,
                                'Component': component,
                                'Content': message
                            })
                    
                    if line_num < 3:
                        tqdm.write(f"  ✓ Total logs: {len(logs)}")
                
                except json.JSONDecodeError as e:
                    if line_num < 10:
                        tqdm.write(f"Warning line {line_num}: JSON error: {e}")
                    continue
                except Exception as e:
                    if line_num < 10:
                        tqdm.write(f"Warning line {line_num}: {type(e).__name__}: {e}")
                    continue
        
        print(f"\nLoaded {len(logs)} valid log entries from {total_lines} lines")
        return logs
    
    def save_templates(self, output_path: str):
        """
        Save log templates to CSV file
        
        Args:
            output_path: Path to save templates CSV
        """
        if self.templates_df is not None:
            self.templates_df.to_csv(output_path, index=False)
            print(f"Templates saved to {output_path}")
        else:
            print("Warning: No templates to save. Run parse() first.")
    
    def get_template_by_id(self, event_id: str) -> str:
        """Get template text by EventId"""
        if self.templates_df is not None:
            row = self.templates_df[self.templates_df['EventId'] == event_id]
            if not row.empty:
                return row.iloc[0]['EventTemplate']
        return None
