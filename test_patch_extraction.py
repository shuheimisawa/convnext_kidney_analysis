import json
import os
import random
import numpy as np
import openslide
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import cv2
import argparse

def load_annotations(geojson_path):
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for feature in data['features']:
        if 'classification' in feature['properties']:
            coords = feature['geometry']['coordinates'][0]
            classification = feature['properties']['classification']['name']
            if classification in ['Normal', 'Sclerotic']:
                polygon = Polygon(coords)
                annotations.append({
                    'polygon': polygon,
                    'class': classification,
                    'bbox': polygon.bounds,
                    'coords': coords
                })
    return annotations

def extract_patch_from_polygon(slide, polygon, coords, target_size=512):
    # Get bounding box dimensions
    minx, miny, maxx, maxy = polygon.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    bbox_center_x = int((minx + maxx) / 2)
    bbox_center_y = int((miny + maxy) / 2)
    
    # Adaptive patch size: ensure complete glomerular capture with 20% padding
    adaptive_size = max(target_size, int(max(bbox_width, bbox_height) * 1.2))
    
    # Center adaptive-sized patch on bounding box center
    x = max(0, bbox_center_x - adaptive_size // 2)
    y = max(0, bbox_center_y - adaptive_size // 2)
    
    # Ensure patch stays within slide bounds
    if x + adaptive_size > slide.dimensions[0]:
        x = slide.dimensions[0] - adaptive_size
    if y + adaptive_size > slide.dimensions[1]:
        y = slide.dimensions[1] - adaptive_size
    
    x = max(0, x)
    y = max(0, y)
    
    # Extract at adaptive size
    patch = slide.read_region((x, y), 0, (adaptive_size, adaptive_size))
    patch = patch.convert('RGB')
    
    # Create overlay showing the annotation on original size
    overlay = patch.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Adjust coordinates relative to patch
    adjusted_coords = [(px - x, py - y) for px, py in coords]
    draw.polygon(adjusted_coords, outline='red', width=3)
    
    # Resize both to target size (512x512) if needed
    if adaptive_size != target_size:
        patch = patch.resize((target_size, target_size), Image.LANCZOS)
        overlay = overlay.resize((target_size, target_size), Image.LANCZOS)
    
    return patch, overlay, (x, y, adaptive_size)

def is_empty_background(patch):
    """Check if patch is mostly white/empty (slide margins)"""
    patch_array = np.array(patch)
    gray = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    return mean_brightness > 200 and std_brightness < 20

def extract_background_patch(slide, annotations, patch_size=512, max_attempts=100):
    for attempt in range(max_attempts):
        x = random.randint(0, slide.dimensions[0] - patch_size)
        y = random.randint(0, slide.dimensions[1] - patch_size)
        
        patch_center = Point(x + patch_size // 2, y + patch_size // 2)
        
        # Check if overlaps with any glomeruli annotations
        overlaps = False
        for ann in annotations:
            if ann['polygon'].contains(patch_center) or ann['polygon'].intersects(
                Polygon([(x, y), (x + patch_size, y), (x + patch_size, y + patch_size), (x, y + patch_size)])
            ):
                overlaps = True
                break
        
        if not overlaps:
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch_rgb = patch.convert('RGB')
            
            # Apply white background filter ONLY to background patches
            if not is_empty_background(patch_rgb):
                return patch_rgb, (x, y)
    
    return None, None

def visualize_patches(patches_data, title_prefix, num_samples=10):
    # Create grid for 10 samples: 2 rows x 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(title_prefix, fontsize=16)
    
    for i, (patch, overlay, class_name, location) in enumerate(patches_data):
        if i >= num_samples:
            break
            
        row1 = i // 5  # First row for originals
        row2 = row1 + 2  # Second row for overlays (2 rows down)
        col = i % 5
        
        # Original patch
        axes[row1, col].imshow(patch)
        if len(location) == 3:  # Has adaptive size info
            axes[row1, col].set_title(f'{class_name}\nLoc: ({location[0]},{location[1]})\nAdaptive: {location[2]}px', fontsize=7)
        else:
            axes[row1, col].set_title(f'{class_name}\nLoc: {location}', fontsize=8)
        axes[row1, col].axis('off')
        
        # Overlay (if available)
        if overlay is not None:
            axes[row2, col].imshow(overlay)
            axes[row2, col].set_title(f'{class_name} - Annotated', fontsize=8)
        else:
            axes[row2, col].imshow(patch)
            axes[row2, col].set_title(f'{class_name} - Background', fontsize=8)
        axes[row2, col].axis('off')
    
    plt.tight_layout()
    filename = f'visualization_outputs/{title_prefix.lower().replace(" ", "_")}_patches.png'
    os.makedirs('visualization_outputs', exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved visualization: {filename}")

def test_patch_extraction():
    # Use first available slide for testing
    slides_dir = "data/raw images"
    annotations_dir = "data/geoJson annotations"
    
    slide_files = [f for f in os.listdir(slides_dir) if f.endswith('.svs')]
    
    if not slide_files:
        print("No SVS files found!")
        return
    
    # Test with first slide
    test_slide = slide_files[0]
    slide_name = os.path.splitext(test_slide)[0]
    geojson_file = f"{slide_name}.geojson"
    
    slide_path = os.path.join(slides_dir, test_slide)
    geojson_path = os.path.join(annotations_dir, geojson_file)
    
    if not os.path.exists(geojson_path):
        print(f"Annotation file not found: {geojson_path}")
        return
    
    print(f"Testing with slide: {test_slide}")
    
    slide = openslide.OpenSlide(slide_path)
    annotations = load_annotations(geojson_path)
    
    print(f"Slide dimensions: {slide.dimensions}")
    print(f"Found {len(annotations)} annotations")
    
    # Count by class
    normal_anns = [ann for ann in annotations if ann['class'] == 'Normal']
    sclerotic_anns = [ann for ann in annotations if ann['class'] == 'Sclerotic']
    
    print(f"Normal: {len(normal_anns)}, Sclerotic: {len(sclerotic_anns)}")
    
    all_patches = []
    
    # Extract 10 normal patches
    for i, ann in enumerate(normal_anns[:10]):
        patch, overlay, location = extract_patch_from_polygon(
            slide, ann['polygon'], ann['coords'], target_size=512
        )
        all_patches.append((patch, overlay, f"Normal {i+1}", location))
    
    # Extract 10 sclerotic patches
    for i, ann in enumerate(sclerotic_anns[:10]):
        patch, overlay, location = extract_patch_from_polygon(
            slide, ann['polygon'], ann['coords'], target_size=512
        )
        all_patches.append((patch, overlay, f"Sclerotic {i+1}", location))
    
    # Extract 10 background patches
    background_patches_collected = 0
    for i in range(50):  # Try more attempts to get 10 good background patches
        bg_patch, location = extract_background_patch(slide, annotations, patch_size=512)
        if bg_patch:
            all_patches.append((bg_patch, None, f"Background {background_patches_collected+1}", location))
            background_patches_collected += 1
            if background_patches_collected >= 10:
                break
    
    slide.close()
    
    # Visualize in groups of 10
    normal_patches = all_patches[0:10]
    sclerotic_patches = all_patches[10:20]
    background_patches = all_patches[20:30]
    
    visualize_patches(normal_patches, "Normal Glomeruli Patches (10 samples)", 10)
    visualize_patches(sclerotic_patches, "Sclerotic Glomeruli Patches (10 samples)", 10) 
    visualize_patches(background_patches, "Background Patches (10 samples)", 10)
    
    print("\nPatch extraction test completed!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    test_patch_extraction()