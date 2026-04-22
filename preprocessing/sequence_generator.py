"""
Sequence Generator for CSCLog
Creates EventSequences from parsed logs using sliding windows
"""

import pandas as pd
import json
from typing import Dict, List, Tuple
from datetime import datetime
import dateutil.parser
from tqdm import tqdm


class SequenceGenerator:
    """Generate EventSequences from parsed logs"""
    
    def __init__(self, window_size: int = 9, session_type: str = "sliding",
                 time_window_seconds: int = 10, write_interval: int = 50000):
        """
        Initialize sequence generator
        
        Args:
            window_size: Sliding window size (6-14)
            session_type: 'sliding' or 'time_window'
            time_window_seconds: Time window size for time-based sessions
            write_interval: Write to disk every N sequences
        """
        self.window_size = window_size
        self.session_type = session_type
        self.time_window_seconds = time_window_seconds
        self.write_interval = write_interval
        
        if not (6 <= window_size <= 14):
            raise ValueError(f"window_size must be between 6 and 14, got {window_size}")
    
    def generate_sequences(self, jsonl_path: str, event_mapping: Dict[int, str],
                          component_ids: List[int], output_path: str,
                          progress_bar: bool = True) -> pd.DataFrame:
        """
        Generate EventSequences from logs
        
        Args:
            jsonl_path: Path to original JSONL file
            event_mapping: Dict mapping log_index -> EventId
            component_ids: List of component IDs for each log
            output_path: Path to save sequences CSV
            progress_bar: Show progress bar
            
        Returns:
            DataFrame with columns: [SessionId, EventSequence, Label]
        """
        print(f"Generating sequences with window_size={self.window_size}, type={self.session_type}...")
        
        # Load logs with timestamps
        logs = self._load_logs_with_metadata(jsonl_path, event_mapping, component_ids, progress_bar)
        
        # Generate sequences based on session type
        if self.session_type == "sliding":
            sequences = self._sliding_window(logs, progress_bar)
        elif self.session_type == "time_window":
            sequences = self._time_window(logs, progress_bar)
        else:
            raise ValueError(f"Unknown session_type: {self.session_type}")
        
        # Convert to DataFrame
        df = pd.DataFrame(sequences, columns=['SessionId', 'EventSequence', 'Label'])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} sequences, saved to {output_path}")
        
        return df
    
    def _load_logs_with_metadata(self, jsonl_path: str, event_mapping: Dict[int, str],
                                 component_ids: List[int], progress_bar: bool = True) -> List[Tuple]:
        """
        Load logs with EventId, ComponentId, and Timestamp
        
        Returns:
            List of (EventId, ComponentId, Timestamp) tuples
        """
        logs = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            iterator = tqdm(f, total=total_lines, desc="Loading logs") if progress_bar else f
            
            for idx, line in enumerate(iterator):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse JSONL
                    # Format: app.kolla: [timestamp, {log_data}]
                    # Format: remote.*: [timestamp, {log_data}]
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            json_part = parts[1].strip()
                            data = json.loads(json_part)
                            if isinstance(data, list) and len(data) >= 2:
                                log_data = data[1]
                                timestamp = log_data.get('timestamp', '')
                                
                                # Skip if no timestamp
                                if not timestamp:
                                    continue
                                
                                # Get EventId and ComponentId
                                if idx in event_mapping:
                                    event_id = event_mapping[idx]
                                    component_id = component_ids[idx] if idx < len(component_ids) else -1
                                    
                                    logs.append((event_id, component_id, timestamp))
                
                except Exception as e:
                    continue
        
        print(f"Loaded {len(logs)} log entries with metadata")
        return logs
                                    logs.append((event_id, component_id, timestamp))
                
                except Exception as e:
                    continue
        
        return logs
    
    def _sliding_window(self, events: List[Tuple], progress_bar: bool = True) -> List[List]:
        """
        Apply sliding window to create sequences
        
        Args:
            events: List of (EventId, ComponentId, Timestamp) tuples
            progress_bar: Show progress bar
            
        Returns:
            List of [SessionId, EventSequence, Label]
        """
        sequences = []
        session_id = 0
        
        total_windows = max(0, len(events) - self.window_size + 1)
        iterator = tqdm(range(total_windows), desc="Creating sequences") if progress_bar else range(total_windows)
        
        for i in iterator:
            window = events[i:i + self.window_size]
            
            # Format as list of tuples (EventId, ComponentId, Timestamp)
            event_sequence = [(eid, cid, ts) for eid, cid, ts in window]
            
            # Default label is 0 (normal)
            label = 0
            
            sequences.append([f"S{session_id:06d}", str(event_sequence), label])
            session_id += 1
        
        return sequences
    
    def _time_window(self, events: List[Tuple], progress_bar: bool = True) -> List[List]:
        """
        Group events by time windows
        
        Args:
            events: List of (EventId, ComponentId, Timestamp) tuples
            progress_bar: Show progress bar
            
        Returns:
            List of [SessionId, EventSequence, Label]
        """
        sequences = []
        session_id = 0
        
        if not events:
            return sequences
        
        # Group by time windows
        current_window = []
        window_start_time = None
        
        iterator = tqdm(events, desc="Creating time-based sequences") if progress_bar else events
        
        for event in iterator:
            event_id, component_id, timestamp_str = event
            
            try:
                event_time = dateutil.parser.parse(timestamp_str)
            except:
                # Skip events with invalid timestamps
                continue
            
            # Start new window if needed
            if window_start_time is None:
                window_start_time = event_time
                current_window = [event]
            else:
                time_diff = (event_time - window_start_time).total_seconds()
                
                if time_diff <= self.time_window_seconds:
                    # Add to current window
                    current_window.append(event)
                else:
                    # Save current window and start new one
                    if len(current_window) >= self.window_size:
                        event_sequence = [(eid, cid, ts) for eid, cid, ts in current_window]
                        sequences.append([f"S{session_id:06d}", str(event_sequence), 0])
                        session_id += 1
                    
                    # Start new window
                    window_start_time = event_time
                    current_window = [event]
        
        # Add last window
        if len(current_window) >= self.window_size:
            event_sequence = [(eid, cid, ts) for eid, cid, ts in current_window]
            sequences.append([f"S{session_id:06d}", str(event_sequence), 0])
        
        return sequences
    
    def save_sequences(self, sequences: pd.DataFrame, output_path: str):
        """
        Save sequences to CSV file
        
        Args:
            sequences: DataFrame with sequences
            output_path: Path to save CSV
        """
        sequences.to_csv(output_path, index=False)
        print(f"Sequences saved to {output_path}")
