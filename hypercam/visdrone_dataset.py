"""
VisDrone Dataset Module - Optimized for large datasets
"""
import os
import gc
import time
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VisDroneDataset(Dataset):
    """
    VisDrone Dataset for object detection
    """
    def __init__(self, img_root, ann_root, patch_size=64, max_samples=None, 
                 augment=False, ignore_class_0=True, min_box_size=20, debug=False):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []
        self.ignore_class_0 = ignore_class_0
        self.min_box_size = min_box_size
        self.debug = debug
        self.max_samples = max_samples
        self.sample_count = 0
        
        print(f"Scanning dataset in {img_root}...")
        seq_dirs = [d for d in sorted(os.listdir(img_root)) 
                   if os.path.isdir(os.path.join(img_root, d))]
        
        # Process sequences with progress bar
        for seq_name in tqdm(seq_dirs, desc="Processing sequences"):
            seq_img_dir = os.path.join(img_root, seq_name)
            ann_path = os.path.join(ann_root, f"{seq_name}.txt")
            
            if not os.path.exists(ann_path):
                if self.debug:
                    print(f"Warning: Annotation file not found: {ann_path}")
                continue
            
            # Process annotations
            frame_dict = {}
            try:
                with open(ann_path, "r") as f:
                    for line in f:
                        try:
                            vals = line.strip().split(',')
                            if len(vals) < 8:
                                continue
                            
                            frame_id = int(vals[0])
                            x, y, w, h = map(float, vals[2:6])
                            cat = int(vals[7])
                            
                            if self.ignore_class_0 and cat == 0:
                                continue
                            if w <= 0 or h <= 0 or w < self.min_box_size or h < self.min_box_size:
                                continue
                            
                            if frame_id not in frame_dict:
                                frame_dict[frame_id] = []
                            frame_dict[frame_id].append([x, y, w, h, cat])
                        except (ValueError, IndexError) as e:
                            if self.debug:
                                print(f"Warning: Error parsing annotation line: {line.strip()}, Error: {e}")
                            continue
            except Exception as e:
                print(f"Error reading {ann_path}: {e}")
                continue
            
            # Process images
            try:
                img_files = [f for f in os.listdir(seq_img_dir) if f.endswith((".jpg", ".png"))]
                for img_name in tqdm(sorted(img_files), desc=f"Processing {seq_name}", leave=False):
                    try:
                        frame_id = int(os.path.splitext(img_name)[0])
                        img_path = os.path.join(seq_img_dir, img_name)
                        
                        annos = frame_dict.get(frame_id, [])
                        if not annos:
                            continue
                        
                        self.samples.append((img_path, annos))
                        
                        if self.max_samples and len(self.samples) >= self.max_samples:
                            break
                    except (ValueError, Exception) as e:
                        if self.debug:
                            print(f"Warning: Error processing {img_name}: {e}")
                        continue
                    
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
                    
            except Exception as e:
                print(f"Error processing sequence {seq_name}: {e}")
                continue
        
        # Print dataset statistics
        if self.debug and self.samples:
            self._print_statistics()
        
        print(f"\nLoaded {len(self.samples)} frames with valid annotations")
        
    def _print_statistics(self):
        """Print dataset statistics"""
        class_counts = {}
        total_boxes = 0
        
        for _, annos in self.samples:
            for anno in annos:
                cat = int(anno[4])
                class_counts[cat] = class_counts.get(cat, 0) + 1
                total_boxes += 1
        
        print(f"\nDataset Statistics:")
        print(f"Total frames: {len(self.samples)}")
        print(f"Total bounding boxes: {total_boxes}")
        print("\nClass distribution:")
        for cat in sorted(class_counts.keys()):
            print(f"  Class {cat}: {class_counts[cat]} boxes ({class_counts[cat]/total_boxes*100:.1f}%)")
    
    def __getitem__(self, idx):
        if self.max_samples and self.sample_count >= self.max_samples:
            return [], [], []
            
        img_path, annos = self.samples[idx]
        
        try:
            # Use a context manager to ensure the file is properly closed
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_array = np.array(img)
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            return [], [], []
        
        patches = []
        labels = []
        
        for anno in annos:
            if self.max_samples and self.sample_count >= self.max_samples:
                break
                
            try:
                x, y, w, h, cat = anno
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # First calculate all coordinates with bounds checking
                h_img, w_img = img_array.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                
                # Skip if the patch is too small after bounds checking
                if (x2 - x1) < 10 or (y2 - y1) < 10:  # Minimum size threshold
                    continue
                    
                try:
                    # Extract the patch
                    patch = img_array[y1:y2, x1:x2]
                    
                    # Skip if the patch is too small after bounds checking
                    if (x2 - x1) < 10 or (y2 - y1) < 10:  # Minimum size threshold
                        continue
                        
                    patch = img_array[y1:y2, x1:x2]
                    
                    # Skip if patch is empty
                    if patch.size == 0:
                        continue
                        
                    # Convert to PIL Image for resizing
                    patch_img = Image.fromarray(patch)
                    patch_img = patch_img.resize((self.patch_size, self.patch_size))
                    patch_array = np.array(patch_img)
                    
                    patches.append(patch_array)
                    labels.append(cat)
                    self.sample_count += 1
                    
                except Exception as e:
                    if self.debug:
                        print(f"Error processing patch: {e}")
                    continue
                    
            except Exception as e:
                if self.debug:
                    print(f"Error processing patch in {img_path}: {e}")
                continue
        
        # If no valid patches were found, return empty lists
        if not patches:
            return [], [], []
            
        return patches, labels, [img_path] * len(patches)
    
    def __len__(self):
        return len(self.samples)
