"""
Component Extractor for CSCLog
Extracts and maps component information from log metadata
"""

import json
from typing import Dict, List, Tuple


class ComponentExtractor:
    """Extract and map component information from logs"""
    
    def __init__(self):
        """Initialize component mapping"""
        self.component_map: Dict[str, int] = {}
        self.next_id: int = 0
    
    def extract_components(self, logs: List[Dict]) -> Tuple[List[int], Dict[str, int]]:
        """
        Extract component IDs from log entries
        
        Args:
            logs: List of log dictionaries with metadata
            
        Returns:
            component_ids: List of component IDs for each log
            component_map: Dict mapping component_name -> component_id
        """
        component_ids = []
        
        for log in logs:
            component_name = self._get_component_name(log)
            component_id = self._get_or_create_id(component_name)
            component_ids.append(component_id)
        
        print(f"Extracted {len(self.component_map)} unique components")
        
        return component_ids, self.component_map
    
    def _get_component_name(self, log_entry: Dict) -> str:
        """
        Extract component name from log entry
        Priority: service > host > default
        
        Args:
            log_entry: Log dictionary
            
        Returns:
            Component name string
        """
        # Priority 1: service field
        if 'service' in log_entry and log_entry['service']:
            return log_entry['service']
        
        # Priority 2: host field
        if 'host' in log_entry and log_entry['host']:
            return log_entry['host']
        
        # Priority 3: Component field (from Drain parser output)
        if 'Component' in log_entry and log_entry['Component']:
            return log_entry['Component']
        
        # Default: unknown
        return "unknown"
    
    def _get_or_create_id(self, component_name: str) -> int:
        """
        Get existing component ID or create new one
        
        Args:
            component_name: Name of component
            
        Returns:
            Component ID (integer)
        """
        # Handle missing/unknown components
        if not component_name or component_name == "unknown":
            return -1
        
        # Return existing ID if already mapped
        if component_name in self.component_map:
            return self.component_map[component_name]
        
        # Create new ID
        new_id = self.next_id
        self.component_map[component_name] = new_id
        self.next_id += 1
        
        return new_id
    
    def save_mapping(self, output_path: str):
        """
        Save component mapping to JSON file
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.component_map, f, indent=2)
        print(f"Component mapping saved to {output_path}")
        print(f"Total components: {len(self.component_map)}")
        
        # Print component statistics
        print("\nComponent distribution:")
        for comp_name, comp_id in sorted(self.component_map.items(), key=lambda x: x[1]):
            print(f"  {comp_id}: {comp_name}")
    
    def get_component_name(self, component_id: int) -> str:
        """
        Get component name by ID
        
        Args:
            component_id: Component ID
            
        Returns:
            Component name or "unknown"
        """
        if component_id == -1:
            return "unknown"
        
        for name, cid in self.component_map.items():
            if cid == component_id:
                return name
        
        return "unknown"
    
    def get_num_components(self) -> int:
        """Get total number of unique components"""
        return len(self.component_map)
