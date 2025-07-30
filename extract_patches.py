import json
import os
import random
import numpy as np
import openslide
from PIL import Image
import cv2
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import argparse

def load_annotations(geojson_path):
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for feature in data['features']:
        if 'classification' in feature['properties']:
            coords = feature['geometry']['coordinates'][0]  # Get first ring
            classification = feature['properties']['classification']['name']
            if classification in ['Normal', 'Sclerotic']:
                # Ensure coords is a list of [x, y] pairs
                if len(coords) > 0 and isinstance(coords[0], list) and len(coords[0]) == 2:
                    polygon = Polygon(coords)
                    annotations.append({
                        'polygon': polygon,
                        'class': classification,
                        'coords': coords
                    })
    return annotations

def extract_patch_from_polygon(slide, polygon, coords, target_size=512):
    minx, miny, maxx, maxy = polygon.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny
    bbox_center_x = int((minx + maxx) / 2)
    bbox_center_y = int((miny + maxy) / 2)
    
    adaptive_size = max(target_size, int(max(bbox_width, bbox_height) * 1.2))
    
    x = max(0, bbox_center_x - adaptive_size // 2)
    y = max(0, bbox_center_y - adaptive_size // 2)
    
    if x + adaptive_size > slide.dimensions[0]:
        x = slide.dimensions[0] - adaptive_size
    if y + adaptive_size > slide.dimensions[1]:
        y = slide.dimensions[1] - adaptive_size
    
    x = max(0, x)
    y = max(0, y)
    
    patch = slide.read_region((x, y), 0, (adaptive_size, adaptive_size))
    patch = patch.convert('RGB')
    
    if adaptive_size != target_size:
        patch = patch.resize((target_size, target_size), Image.LANCZOS)
    
    return patch

def is_empty_background(patch):
    patch_array = np.array(patch)
    gray = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) > 200 and np.std(gray) < 20

def extract_background_patch(slide, annotations, patch_size=512, max_attempts=100):
    for _ in range(max_attempts):
        x = random.randint(0, slide.dimensions[0] - patch_size)
        y = random.randint(0, slide.dimensions[1] - patch_size)
        
        patch_center = Point(x + patch_size // 2, y + patch_size // 2)
        
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
            
            if not is_empty_background(patch_rgb):
                return patch_rgb
    
    return None

def extract_patches_from_slide(slide_path, geojson_path, output_dir, target_size=512):
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    slide = openslide.OpenSlide(slide_path)
    annotations = load_annotations(geojson_path)
    
    patch_paths = {'normal': [], 'sclerotic': [], 'background': []}
    
    # Extract glomeruli patches
    for i, ann in enumerate(annotations):
        patch = extract_patch_from_polygon(slide, ann['polygon'], ann['coords'], target_size)
        if patch:
            class_name = ann['class'].lower()
            patch_path = os.path.join(output_dir, class_name, f"{slide_name}_{class_name}_{i:03d}.png")
            os.makedirs(os.path.dirname(patch_path), exist_ok=True)
            patch.save(patch_path)
            patch_paths[class_name].append(patch_path)
    
    # Extract background patches
    for i in range(len(annotations)):
        bg_patch = extract_background_patch(slide, annotations, target_size)
        if bg_patch:
            patch_path = os.path.join(output_dir, 'background', f"{slide_name}_background_{i:03d}.png")
            os.makedirs(os.path.dirname(patch_path), exist_ok=True)
            bg_patch.save(patch_path)
            patch_paths['background'].append(patch_path)
    
    slide.close()
    return patch_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--output_dir', default='patches')
    parser.add_argument('--target_per_class', type=int, default=1013)
    args = parser.parse_args()
    
    slides_dir = "data/raw images"
    annotations_dir = "data/geoJson annotations"
    
    all_patches = {'normal': [], 'sclerotic': [], 'background': []}
    
    slide_files = [f for f in os.listdir(slides_dir) if f.endswith('.svs')]
    
    for slide_file in tqdm(slide_files, desc="Processing slides"):
        slide_name = os.path.splitext(slide_file)[0]
        geojson_file = f"{slide_name}.geojson"
        
        slide_path = os.path.join(slides_dir, slide_file)
        geojson_path = os.path.join(annotations_dir, geojson_file)
        
        if os.path.exists(geojson_path):
            slide_patches = extract_patches_from_slide(
                slide_path, geojson_path, args.output_dir, args.target_size
            )
            for class_name, patches in slide_patches.items():
                all_patches[class_name].extend(patches)
    
    print(f"Extracted patches: Normal={len(all_patches['normal'])}, Sclerotic={len(all_patches['sclerotic'])}, Background={len(all_patches['background'])}")
    
    # Balance dataset
    for class_name in all_patches:
        random.shuffle(all_patches[class_name])
        all_patches[class_name] = all_patches[class_name][:args.target_per_class]
    
    # Create training file
    with open('train_patches.txt', 'w') as f:
        for patch_path in all_patches['normal']:
            f.write(f"{patch_path},0\n")
        for patch_path in all_patches['sclerotic']:
            f.write(f"{patch_path},1\n")
        for patch_path in all_patches['background']:
            f.write(f"{patch_path},2\n")
    
    balanced_count = min(len(all_patches[cls]) for cls in all_patches)
    print(f"Created balanced dataset: {balanced_count} per class")
    print("Patch list saved to train_patches.txt")

if __name__ == "__main__":
    main()