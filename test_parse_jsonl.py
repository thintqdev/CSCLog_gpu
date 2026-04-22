"""Quick test to debug JSONL parsing"""

import json

# Test parsing first line
with open('dataset/data_full.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        
        print(f"\n=== Line {i} ===")
        print(f"Raw: {line[:150]}")
        
        line = line.strip()
        
        # Check colon
        if ':' not in line:
            print("ERROR: No colon!")
            continue
        
        # Split
        parts = line.split(':', 1)
        print(f"Parts: {len(parts)}")
        print(f"Prefix: {parts[0]}")
        print(f"JSON part: {parts[1][:100]}...")
        
        # Parse JSON
        try:
            data = json.loads(parts[1].strip())
            print(f"Data type: {type(data)}")
            print(f"Data length: {len(data) if isinstance(data, list) else 'N/A'}")
            
            if isinstance(data, list) and len(data) >= 2:
                log_data = data[1]
                print(f"Log data keys: {log_data.keys()}")
                print(f"Timestamp: {log_data.get('timestamp')}")
                print(f"Service: {log_data.get('service')}")
                print(f"Message: {log_data.get('message')}")
                
                # Check if message and timestamp exist
                if log_data.get('message') and log_data.get('timestamp'):
                    print("✓ VALID LOG ENTRY")
                else:
                    print("✗ MISSING message or timestamp")
        except Exception as e:
            print(f"ERROR: {e}")
