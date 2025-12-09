import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from drone import HDCWirelessDetector, BaselineWirelessDetector
import torch

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _patch_torch_load():
    original_load = torch.load
    def patched_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(f, *args, **kwargs)
    torch.load = patched_load

_patch_torch_load()

# VisDrone class names (10 classes)
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# Colors for each class (BGR format for OpenCV)
CLASS_COLORS = [
    (255, 0, 0),      # pedestrian - blue
    (0, 255, 0),      # people - green
    (0, 0, 255),      # bicycle - red
    (255, 255, 0),    # car - cyan
    (255, 0, 255),    # van - magenta
    (0, 255, 255),    # truck - yellow
    (128, 0, 128),    # tricycle - purple
    (255, 128, 0),    # awning-tricycle - orange
    (128, 128, 0),    # bus - olive
    (0, 128, 128),    # motor - teal
]


def draw_detections(image: np.ndarray, detections: List[Dict],
                   title: str = "", confidence_threshold: float = 0.3) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    if title:
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

    for det in detections:
        if det['confidence'] < confidence_threshold:
            continue

        bbox = det['bbox']
        cls = det['class']
        conf = det['confidence']

        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}'
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (128, 128, 128)

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f'{class_name}: {conf:.2f}'
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 8), (x1 + label_w + 8, y1), color, -1)

        cv2.putText(img, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

    count_text = f'Detections: {len([d for d in detections if d["confidence"] >= confidence_threshold])}'
    cv2.putText(img, count_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 255, 0), 2)

    return img


def save_comparison_visualization(
    sequence_path: str,
    frame_numbers: List[int],
    model_path: str,
    output_dir: str,
    yolo_interval: int = 10,
    confidence_threshold: float = 0.3
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    yolo_model = YOLO(model_path)

    baseline_detector = BaselineWirelessDetector(yolo_model)
    hdc_detector = HDCWirelessDetector(yolo_model, yolo_interval=yolo_interval)

    sequence_name = os.path.basename(sequence_path)
    print(f"\nProcessing: {sequence_name} ({len(frame_numbers)} frames)")

    for frame_num in frame_numbers:
        frame_path = os.path.join(sequence_path, f'{frame_num:07d}.jpg')
        if not os.path.exists(frame_path):
            print(f"Frame {frame_num} not found")
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame {frame_num}")
            continue

        h, w = frame.shape[:2]

        baseline_detections = baseline_detector.detect(frame, frame_path, original_size=(h, w))

        hdc_detector.tracker.frame_count = frame_num - 1
        hdc_detections, _ = hdc_detector.detect(frame, frame_path)

        img_original = frame.copy()
        cv2.putText(img_original, "Original", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2)

        img_baseline = draw_detections(frame, baseline_detections,
                                       "Baseline (Every Frame)", confidence_threshold)

        img_hdc = draw_detections(frame, hdc_detections,
                                 f"HDC (Interval={yolo_interval})", confidence_threshold)

        comparison = np.hstack([img_original, img_baseline, img_hdc])

        output_prefix = f"{sequence_name}_frame{frame_num:04d}"
        cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_original.jpg"), img_original)
        cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_baseline.jpg"), img_baseline)
        cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_hdc.jpg"), img_hdc)
        cv2.imwrite(os.path.join(output_dir, f"{output_prefix}_comparison.jpg"), comparison)

        print(f"Frame {frame_num}: {len(baseline_detections)} baseline, {len(hdc_detections)} HDC")

    print(f"\nSaved to: {output_dir}/")


def save_detection_sequence(
    sequence_path: str,
    start_frame: int,
    num_frames: int,
    model_path: str,
    output_dir: str,
    detector_type: str = 'hdc',
    yolo_interval: int = 10,
    confidence_threshold: float = 0.3
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    yolo_model = YOLO(model_path)

    if detector_type == 'baseline':
        detector = BaselineWirelessDetector(yolo_model)
        title_prefix = "Baseline"
    else:
        detector = HDCWirelessDetector(yolo_model, yolo_interval=yolo_interval)
        title_prefix = f"HDC (Int={yolo_interval})"

    sequence_name = os.path.basename(sequence_path)
    print(f"\nProcessing: {sequence_name} ({num_frames} frames from {start_frame})")

    for i in range(num_frames):
        frame_num = start_frame + i
        frame_path = os.path.join(sequence_path, f'{frame_num:07d}.jpg')

        if not os.path.exists(frame_path):
            print(f"Frame {frame_num} not found")
            break

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Failed to load frame {frame_num}")
            break

        h, w = frame.shape[:2]

        if detector_type == 'baseline':
            detections = detector.detect(frame, frame_path, original_size=(h, w))
        else:
            detections, _ = detector.detect(frame, frame_path)

        title = f"{title_prefix} - Frame {frame_num}"
        img_annotated = draw_detections(frame, detections, title, confidence_threshold)

        output_name = f"{sequence_name}_{detector_type}_frame{frame_num:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, output_name), img_annotated)

        print(f"Frame {frame_num}: {len(detections)} detections")

    print(f"\nSaved to: {output_dir}/")


if __name__ == '__main__':
    MODEL_PATH = "yolo_nano.pt"
    DATA_PATH = "data/VisDrone2019-VID"
    OUTPUT_DIR = "visualizations"

    print("="*80)
    print("Baseline vs HDC Comparison")
    print("="*80)

    save_comparison_visualization(
        sequence_path=f"{DATA_PATH}/sequences/uav0000013_00000_v",
        frame_numbers=[1, 50, 100, 150, 200],
        model_path=MODEL_PATH,
        output_dir=f"{OUTPUT_DIR}/comparison",
        yolo_interval=10,
        confidence_threshold=0.3
    )

    print("\n" + "="*80)
    print("HDC Tracking Sequence")
    print("="*80)

    save_detection_sequence(
        sequence_path=f"{DATA_PATH}/sequences/uav0000013_00000_v",
        start_frame=50,
        num_frames=10,
        model_path=MODEL_PATH,
        output_dir=f"{OUTPUT_DIR}/hdc_sequence",
        detector_type='hdc',
        yolo_interval=10,
        confidence_threshold=0.3
    )

    print("\n" + "="*80)
    print("High-Density Scene")
    print("="*80)

    save_comparison_visualization(
        sequence_path=f"{DATA_PATH}/sequences/uav0000072_04488_v",
        frame_numbers=[1, 20, 40, 60, 80],
        model_path=MODEL_PATH,
        output_dir=f"{OUTPUT_DIR}/high_density",
        yolo_interval=10,
        confidence_threshold=0.3
    )

    print("\n" + "="*80)
    print("Complete")
    print("="*80)
