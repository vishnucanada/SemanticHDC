# SemanticHDC: Hyperdimensional Computing for Semantic Communication in UAVs

A novel framework combining **Hyperdimensional Computing (HDC)** with **Semantic Communication** for energy-efficient object detection and tracking on resource-constrained UAV platforms.

## Overview

This research project demonstrates how HDC can achieve:
- **28% higher accuracy** (73.99% vs 57.63% mAP@50) compared to per-frame YOLO baselines
- **54-82% energy reduction** through semantic packet transmission
- **5-14× packet compression** by transmitting semantic object representations instead of raw frames

The system alternates between YOLO object detection at periodic intervals and lightweight HDC-based tracking in between, transmitting compact semantic packets (~0.5-1.7 KB) instead of full JPEG frames (~7-9 KB).

## Dataset Setup

### VisDrone2019-VID Dataset

1. **Download the dataset:**
   - Visit the [VisDrone Dataset GitHub](https://github.com/VisDrone/VisDrone-Dataset)
   - Navigate to **Task 2: Object Detection in Videos** (VisDrone2019-VID)
   - Download:
     - `sequences/` folder (video frames)
     - `annotations/` folder (ground truth bounding boxes)

2. **Organize the dataset:**
   ```
   SemanticHDC/
   └── data/
       └── VisDrone2019-VID/
           ├── sequences/
           │   ├── uav0000013_00000_v/
           │   │   ├── 0000001.jpg
           │   │   ├── 0000002.jpg
           │   │   └── ...
           │   ├── uav0000138_00000_v/
           │   ├── uav0000218_00001_v/
           │   └── ...
           └── annotations/
               ├── uav0000013_00000_v.txt
               ├── uav0000138_00000_v.txt
               ├── uav0000218_00001_v.txt
               └── ...
   ```

3. **Dataset Details:**
   - 56 video sequences
   - 11 object categories (person, car, van, truck, bicycle, awning-tricycle, bus, motor, tricycle, pedestrian, people)
   - Captured from various altitudes (15-120m) and scenarios (urban, highway, parking lots)
   - Frame resolution: 1920×1080 (downsampled to 640×640 for experiments)

### Annotation Format
Each annotation file contains bounding boxes in the format:
```
<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

## Model Setup

### YOLOv8-Nano Pretrained Model

1. **Download from Hugging Face:**
   ```bash
   wget https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/yolov8n_visdrone.pt -O yolo_nano.pt
   ```

   Or download manually:
   - Visit: [https://huggingface.co/mshamrai/yolov8n-visdrone](https://huggingface.co/mshamrai/yolov8n-visdrone)
   - Download `yolov8n_visdrone.pt`
   - Rename to `yolo_nano.pt` and place in project root

2. **Model Details:**
   - Architecture: YOLOv8-nano
   - Fine-tuned on: VisDrone dataset
   - Baseline mAP@50: 57.63%
   - Parameters: ~3.2M (optimized for edge deployment)
   - Inference time: ~25-40ms on desktop GPU

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 16GB+ RAM recommended for dataset processing

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `ultralytics` - YOLOv8 implementation
- `sionna` - 3GPP channel simulation
- `codecarbon` - Energy measurement
- `opencv-python` - Image processing
- `torch` - Deep learning backend
- `numpy`, `scipy` - Numerical operations

## Repository Structure

```
SemanticHDC/
├── data/                           # VisDrone dataset (please install locally, too large for github)
│   └── VisDrone2019-VID/
│       ├── sequences/
│       └── annotations/
├── hypercam/                       # OLD Hypercam HDC Method
│   ├── Encoder.py                 
│   ├── BinarySplatterCode.py      
│   └── visdrone_dataset.py        
├── communication.py                # Semantic packet transmission
├── drone.py                        # HDC tracker & YOLO integration
├── resource_monitoring.py          # Energy profiling
├── main.py                         # Main experiment runner
├── yolo_nano.pt                    # YOLOv8-nano model (download separately)
```

## Code Overview

### Core Modules

#### `communication.py` - Wireless Channel Simulation & Packet Transmission
Implements the wireless communication layer using the **Sionna library**, NVIDIA's open-source framework for link-level simulations of next-generation wireless systems. Sionna provides GPU-accelerated 3GPP channel models (UMi, UMa) with realistic path loss, shadow fading, and Doppler effects.

**Key Classes:**
- `Drone3GPPChannel`: Simulates UAV-to-ground station wireless link using 3GPP TR 38.901 channel models
- `FrameTransmitter3GPP`: Handles JPEG frame compression and transmission over the simulated channel
- `HDCStateTransmitter3GPP`: Transmits compact semantic packets containing HDC-encoded object states
- Utility functions: `calculate_psnr()`, `calculate_ssim()` for measuring received signal quality

#### `drone.py` - Detection Strategies & HDC Tracking
Implements the three detection/tracking strategies (Baseline, Interval, HDC Hybrid) and the core HDC tracking logic.

**Key Classes:**
- `Sionna3GPPDroneChannel`: Configures OFDM resource grid and channel parameters for UAV scenarios
- `HDCTracker`: Implements hyperdimensional computing-based object tracking between YOLO frames
  - Encodes objects using appearance, color, and location hypervectors
  - Performs cosine similarity matching for frame-to-frame association
  - Updates bounding boxes based on HDC spatial reasoning
- `BaselineWirelessDetector`: Runs YOLO on every frame (baseline strategy)
- `BaselineIntervalDetector`: Runs YOLO every N frames, repeats previous detections (interval strategy)
- `HDCWirelessDetector`: Alternates between YOLO and HDC tracking (hybrid strategy)

#### `main.py` - Experiment Orchestration & Evaluation
Orchestrates end-to-end experiments, manages dataset loading, runs evaluations, and computes metrics. Uses **CodeCarbon** to measure energy consumption by tracking CPU/GPU power draw during code execution, then projects these measurements to embedded platform profiles.


#### `resource_monitoring.py` - Energy Profiling
Tracks computational operations (YOLO inference, HDC encoding, transmission) and estimates energy consumption using CodeCarbon measurements combined with Crazyflie AI-Deck power profiles.


### Key Libraries

**Sionna** ([GitHub](https://github.com/NVlabs/sionna)): NVIDIA's TensorFlow-based library for physical-layer research, providing GPU-accelerated 3GPP channel models, LDPC coding, OFDM modulation, and MIMO processing. Used to simulate realistic UAV-to-ground wireless channels with Doppler, fading, and path loss.

**CodeCarbon** ([Docs](https://codecarbon.io/)): Tracks energy consumption and carbon emissions of Python code execution by monitoring CPU/GPU power via hardware interfaces (Intel RAPL, NVIDIA-SMI). Provides process-level energy measurements that we project to embedded platform power profiles.

## Usage

### Run Experiments

**Quick test (single sequence):**
```bash
python main.py --quick-test
```

**Full evaluation (all scenarios):**
```bash
python main.py
```

### Configuration

Edit scenarios in `main.py` (lines 875-930):
```python
scenarios = [
    {
        'name': 'scenario_A_sparse_highway',
        'sequences': ['uav0000218_00001_v'],
        'params': {
            'drone_height_m': 20.0,
            'drone_velocity_ms': 2.0,
            'horizontal_dist_m': 50.0,
            # ... other parameters
        }
    },
    # Add custom scenarios...
]
```

### Output

Results are saved to `results/` directory:
```json
{
  "experiment_name": "HDC_Hybrid_scenario_A",
  "accuracy": {
    "mAP@50": 73.99,
    "precision": 0.82,
    "recall": 0.76
  },
  "transmission": {
    "avg_packet_size_kb": 0.52,
    "total_packets": 269
  },
  "efficiency": {
    "energy_mwh": 86.83,
    "energy_savings_vs_baseline": "82.3%"
  }
}
```

## Hardware Reference

Energy estimates are projected to the **Crazyflie 2.1 AI-Deck** platform:
- **Processor:** GAP8 RISC-V (250 MHz, 8-core)
- **WiFi:** ESP32 (2.4 GHz 802.11n)
- **Camera:** Himax HM01B0 (324×324 grayscale)
- **Battery:** 240 mAh @ 3.7V (0.888 Wh)
- **Documentation:** [Bitcraze Crazyflie Datasheet](https://www.bitcraze.io/documentation/hardware/crazyflie_2_1_plus/crazyflie_2_1_plus-datasheet.pdf)

## Experimental Results

| Scenario | Strategy | mAP@50 | Packet Size | Energy (mWh) | Savings |
|----------|----------|--------|-------------|--------------|---------|
| S1 (Sparse) | Baseline | 57.63% | 7.24 KB | 491.40 | - |
| | Interval | 33.44% | 7.24 KB | 73.29 | -85.1% |
| | **HDC Hybrid** | **73.99%** | **0.52 KB** | **86.83** | **-82.3%** |
| S2 (Dense) | Baseline | 57.63% | 8.91 KB | 995.53 | - |
| | Interval | 33.44% | 8.91 KB | 237.86 | -76.1% |
| | **HDC Hybrid** | **73.99%** | **1.67 KB** | **459.06** | **-53.9%** |



## License

This project is for academic research purposes. The VisDrone dataset and YOLOv8 model have their own respective licenses.

## Contact

- Vishnu Garigipati - v_garigi@live.concordia.ca

Gina Cody School of Engineering and Computer Science
Concordia University, Montreal, Canada
