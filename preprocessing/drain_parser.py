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
                    # Parse JSONL line
                    # Format: app.kolla: [timestamp, {log_data}]
                    # Format: remote.*: [timestamp, {log_data}]
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Debug first 3 lines (will show even with tqdm)
                    if line_num < 3:
                        tqdm.write(f"Debug line {line_num}: Processing...")
                    
                    # Split by first colon to get prefix and data
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            prefix = parts[0].strip()  # app.kolla or remote.*
                            json_part = parts[1].strip()
                            
                            # Parse the array [timestamp, {data}]
                            data = json.loads(json_part)
                            if isinstance(data, list) and len(data) >= 2:
                                log_data = data[1]
                                
                                # Extract fields with fallbacks
                                timestamp = log_data.get('timestamp', '')
                                level = log_data.get('level', 'INFO')
                                
                                # Component priority: service > host > prefix > unknown
                                component = log_data.get('service')
                                if not component:
                                    component = log_data.get('host')
                                if not component:
                                    # Use prefix as component (app.kolla -> kolla, remote.* -> remote)
                                    if '.' in prefix:
                                        component = prefix.split('.', 1)[1] if len(prefix.split('.')) > 1 else prefix
                                    else:
                                        component = prefix
                                if not component:
                                    component = 'unknown'
                                
                                message = log_data.get('message', '')
                                
                                # Debug first few entries
                                if line_num < 3:
                                    tqdm.write(f"  timestamp={timestamp}, component={component}, message={message[:30]}")
                                
                                # Skip empty messages
                                if not message or not timestamp:
                                    if line_num < 3:
                                        tqdm.write(f"  SKIPPED: empty message or timestamp")
                                    continue
                                
                                logs.append({
                                    'Timestamp': timestamp,
                                    'Level': level,
                                    'Component': component,
                                    'Content': message
                                })
                                
                                if line_num < 3:
                                    tqdm.write(f"  ✓ Added to logs (total: {len(logs)})")
                
                except json.JSONDecodeError as e:
                    # Only print first few errors to avoid spam
                    if line_num < 10:
                        tqdm.write(f"Warning: Skipping malformed JSON at line {line_num + 1}: {e}")
                    continue
                except Exception as e:
                    if line_num < 10:
                        tqdm.write(f"Warning: Error processing line {line_num + 1}: {e}")
                    continue
        
        print(f"Loaded {len(logs)} valid log entries from {total_lines} lines")
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
