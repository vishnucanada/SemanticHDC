import os
import numpy as np
from PIL import Image
import time
import tqdm
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
from copy import deepcopy
from scipy import ndimage

load_dotenv()


class VisDroneDataset(Dataset):
    
    def __init__(self, img_root, ann_root, patch_size=32, max_samples=None, 
                 augment=False, ignore_class_0=True, min_box_size=20, debug=False):
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []
        self.ignore_class_0 = ignore_class_0
        self.min_box_size = min_box_size
        self.debug = debug
        
        for seq_name in sorted(os.listdir(img_root)):
            seq_img_dir = os.path.join(img_root, seq_name)
            ann_path = os.path.join(ann_root, f"{seq_name}.txt")
            
            if not os.path.isdir(seq_img_dir) or not os.path.exists(ann_path):
                continue
            
            frame_dict = {}
            with open(ann_path, "r") as f:
                for line in f:
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
            
            for img_name in sorted(os.listdir(seq_img_dir)):
                if not img_name.endswith(".jpg"):
                    continue
                
                frame_id = int(os.path.splitext(img_name)[0])
                img_path = os.path.join(seq_img_dir, img_name)
                
                annos = frame_dict.get(frame_id, [])
                if len(annos) == 0:
                    continue
                
                self.samples.append((img_path, annos))
                
                if max_samples and len(self.samples) >= max_samples:
                    break
            
            if max_samples and len(self.samples) >= max_samples:
                break
        
        # Debug: print class distribution
        if self.debug:
            class_counts = {}
            for _, annos in self.samples:
                for anno in annos:
                    cat = int(anno[4])
                    class_counts[cat] = class_counts.get(cat, 0) + 1
            print(f"\nClass distribution in {img_root}:")
            for cat in sorted(class_counts.keys()):
                print(f"  Class {cat}: {class_counts[cat]} instances")
        
        print(f"Loaded {len(self.samples)} frames")
    
    def __getitem__(self, idx):
        img_path, annos = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        patches = []
        labels = []
        
        for anno in annos:
            x, y, w, h, cat = anno
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            x1, y1 = max(0, x), max(0, y)
            x2 = min(img_array.shape[1], x + w)
            y2 = min(img_array.shape[0], y + h)
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            
            # Extract patch
            patch_rgb = img_array[y1:y2, x1:x2]
            
            # Ensure 3 channels
            if len(patch_rgb.shape) == 2:
                patch_rgb = np.stack([patch_rgb]*3, axis=-1)
            
            # Augment before resize
            if self.augment and np.random.rand() > 0.5:
                patch_rgb = np.fliplr(patch_rgb)
            
            # Resize
            pil_img = Image.fromarray(patch_rgb.astype(np.uint8))
            pil_resized = pil_img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            resized_array = np.array(pil_resized)
            
            # Convert to grayscale
            if len(resized_array.shape) == 3:
                final_patch = np.mean(resized_array, axis=2).astype(np.uint8)
            else:
                final_patch = resized_array.astype(np.uint8)
            
            # Brightness augmentation
            if self.augment and np.random.rand() > 0.7:
                factor = np.random.uniform(0.8, 1.2)
                final_patch = np.clip(final_patch * factor, 0, 255).astype(np.uint8)
            
            patches.append(final_patch)
            labels.append(cat)
        
        return patches, labels, img_path
    
    def __len__(self):
        return len(self.samples)



# ==================== MAIN ====================

if __name__ == "__main__":
    config = {
        "patch_size": 64,
        "hv_length": 10000, 
        "pixel_resolution": 16, 
    }

    
    train_dataset = VisDroneDataset(
        img_root=os.path.join(train_path, 'sequences'),
        ann_root=os.path.join(train_path, 'annotations'),
        patch_size=config["patch_size"],
        max_samples=500,
        augment=True,
        ignore_class_0=True,
        debug=True
    )
    test_dataset = VisDroneDataset(
        img_root=os.path.join(test_path, 'sequences'),
        ann_root=os.path.join(test_path, 'annotations'),
        patch_size=config["patch_size"],
        max_samples=100,
        augment=False,
        ignore_class_0=True,
        debug=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    
    