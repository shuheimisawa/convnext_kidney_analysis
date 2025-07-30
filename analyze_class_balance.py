import json
import os
from collections import defaultdict

def analyze_class_balance():
    annotations_dir = "data/geoJson annotations"
    
    total_counts = defaultdict(int)
    file_counts = {}
    
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.geojson'):
            filepath = os.path.join(annotations_dir, filename)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            file_counts[filename] = defaultdict(int)
            
            for feature in data['features']:
                if 'classification' in feature['properties']:
                    classification = feature['properties']['classification']['name']
                    file_counts[filename][classification] += 1
                    total_counts[classification] += 1
    
    print("=== Overall Class Balance ===")
    for class_name, count in total_counts.items():
        print(f"{class_name}: {count}")
    
    total_samples = sum(total_counts.values())
    print(f"\nTotal annotations: {total_samples}")
    
    if len(total_counts) == 2:
        classes = list(total_counts.keys())
        ratio = total_counts[classes[0]] / total_counts[classes[1]]
        print(f"Ratio {classes[0]}:{classes[1]} = {ratio:.2f}:1")
    
    print(f"\n=== Per-file breakdown ===")
    for filename, counts in sorted(file_counts.items()):
        print(f"{filename}: {dict(counts)}")

if __name__ == "__main__":
    analyze_class_balance()