import os
import time
import json
import warnings
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import torch
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import platform
is_m1_mac = platform.processor() == 'arm' or platform.machine() == 'arm64'

if is_m1_mac:
    num_threads = os.cpu_count() or 8
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    torch.set_num_threads(num_threads)
    print(f"Apple Silicon detected: {num_threads} threads")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS acceleration available")
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU initialization error: {e}")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    torch.set_num_threads(4)

tf.config.optimizer.set_jit(True)

from tqdm import tqdm
from codecarbon import EmissionsTracker
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

from resource_monitoring import ResourceMonitor, EMBEDDED_POWER_PROFILES
from communication import Drone3GPPChannel, FrameTransmitter3GPP, HDCStateTransmitter3GPP
from drone import Sionna3GPPDroneChannel, HDCTracker, BaselineWirelessDetector, BaselineIntervalDetector, HDCWirelessDetector

def _patch_torch_load():
    original_load = torch.load
    def patched_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(f, *args, **kwargs)
    torch.load = patched_load

_patch_torch_load()


@dataclass
class ExperimentResult:
    experiment_name: str
    config: Dict
    transmission: Dict
    accuracy: Dict
    efficiency: Dict
    total_frames: int
    timestamp: str


def load_ground_truth(annotation_file: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes, labels = [], []
    
    if not os.path.exists(annotation_file):
        return np.array([]), np.array([])
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            
            frame = int(parts[0])
            if frame != frame_idx:
                continue
            
            cls = int(parts[7])
            if cls == 0 or cls == 11:
                continue
            cls = cls - 1
            
            x, y, w, h = map(float, parts[2:6])
            if w <= 0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(cls)
    
    return np.array(boxes), np.array(labels)


def extract_frame_number(img_name: str) -> Optional[int]:
    base = os.path.splitext(img_name)[0]
    try:
        return int(base)
    except ValueError:
        return None


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_metrics(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
                   target_cls: np.ndarray, nc: int = 10) -> Dict:
    i = np.argsort(-conf)
    tp, conf, pred_cls, target_cls = tp[i], conf[i], pred_cls[i], target_cls[i]
    
    unique_classes = np.unique(target_cls)
    unique_classes = unique_classes[unique_classes >= 0]
    n_gt = np.bincount(target_cls[target_cls >= 0].astype(int), minlength=nc)
    
    ap = np.zeros(nc)
    p = np.zeros(nc)
    r = np.zeros(nc)
    
    for c in range(nc):
        i_cls = pred_cls == c
        n_pred = i_cls.sum()
        n_gt_c = n_gt[c]
        
        if n_pred == 0 or n_gt_c == 0:
            continue
        
        tp_cls = tp[i_cls]
        tp_cumsum = np.cumsum(tp_cls)
        fp_cumsum = np.cumsum(~tp_cls)
        
        recall_curve = tp_cumsum / (n_gt_c + 1e-16)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap[c] = compute_ap(recall_curve, precision_curve)
        p[c] = precision_curve[-1] if len(precision_curve) > 0 else 0
        r[c] = recall_curve[-1] if len(recall_curve) > 0 else 0
    
    if len(unique_classes) > 0:
        avg_precision = np.nanmean(p[unique_classes])
        avg_recall = np.nanmean(r[unique_classes])
        map50 = np.nanmean(ap[unique_classes])
    else:
        avg_precision, avg_recall, map50 = 0.0, 0.0, 0.0
    
    return {'precision': avg_precision, 'recall': avg_recall, 'map50': map50}


def evaluate_sequence(detector, sequence_path: str, annotation_file: str,
                     transmitter=None, channel_params=None,
                     iou_thres: float = 0.5, resource_monitor: ResourceMonitor = None,
                     embedded_platform: str = 'nvidia_jetson_nano') -> Tuple[Dict, Dict]:

    if resource_monitor is None:
        resource_monitor = ResourceMonitor(embedded_platform=embedded_platform)
    resource_monitor.start()
    
    start_time = time.time()
    
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    
    transmission_stats = {
        'total_transmissions': 0, 'successful': 0, 'failed': 0, 'total_bytes': 0,
        'semantic_similarities': [], 'snr_db_samples': [], 'latency_ms_samples': [],
        'doppler_samples': [], 'ber_samples': [], 'bler_samples': [],
        'psnr_samples': [], 'ssim_samples': [],
        'success_perfect': 0, 'success_good': 0, 'success_usable': 0, 'success_degraded': 0
    }
    
    base_distance = channel_params.get('horizontal_dist_m', 100.0) if channel_params else 100.0
    
    frame_count = 0
    for img_name in sorted(os.listdir(sequence_path)):
        if not img_name.lower().endswith(('.jpg', '.png')):
            continue
        
        frame_idx = extract_frame_number(img_name)
        if frame_idx is None:
            continue
        
        img_path = os.path.join(sequence_path, img_name)
        gt_boxes, gt_labels = load_ground_truth(annotation_file, frame_idx)
        
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        distance_variation = 10 * np.sin(2 * np.pi * frame_count / 100)
        current_distance = base_distance + distance_variation
        
        if isinstance(detector, BaselineWirelessDetector):
            original_size = frame.shape[:2]

            if transmitter is not None:
                with resource_monitor.track_transmission():
                    compressed_data, success, snr_db, latency_ms, received_frame, metrics = transmitter.transmit(
                        frame, current_distance)

                transmission_stats['total_transmissions'] += 1
                transmission_stats['total_bytes'] += len(compressed_data)
                transmission_stats['snr_db_samples'].append(snr_db)
                transmission_stats['latency_ms_samples'].append(latency_ms)
                transmission_stats['ber_samples'].append(metrics.get('ber', 0.0))
                transmission_stats['bler_samples'].append(metrics.get('bler', 0.0))
                transmission_stats['psnr_samples'].append(metrics.get('psnr', 0.0))
                transmission_stats['ssim_samples'].append(metrics.get('ssim', 0.0))

                if metrics.get('success_perfect', False):
                    transmission_stats['success_perfect'] += 1
                if metrics.get('success_good', False):
                    transmission_stats['success_good'] += 1
                if metrics.get('success_usable', False):
                    transmission_stats['success_usable'] += 1
                if metrics.get('success_degraded', False):
                    transmission_stats['success_degraded'] += 1

                with resource_monitor.track_model_computation('yolo'):
                    detections = detector.detect(frame, img_path, original_size=original_size)

                if success:
                    transmission_stats['successful'] += 1
                else:
                    transmission_stats['failed'] += 1
            else:
                with resource_monitor.track_model_computation('yolo'):
                    detections = detector.detect(frame, img_path)
        
        elif isinstance(detector, BaselineIntervalDetector):
            original_size = frame.shape[:2]

            if detector.frame_count % detector.yolo_interval == 0:
                if transmitter is not None:
                    with resource_monitor.track_transmission():
                        compressed_data, success, snr_db, latency_ms, received_frame, metrics = transmitter.transmit(
                            frame, current_distance)

                    transmission_stats['total_transmissions'] += 1
                    transmission_stats['total_bytes'] += len(compressed_data)
                    transmission_stats['snr_db_samples'].append(snr_db)
                    transmission_stats['latency_ms_samples'].append(latency_ms)
                    transmission_stats['ber_samples'].append(metrics.get('ber', 0.0))
                    transmission_stats['bler_samples'].append(metrics.get('bler', 0.0))
                    transmission_stats['psnr_samples'].append(metrics.get('psnr', 0.0))
                    transmission_stats['ssim_samples'].append(metrics.get('ssim', 0.0))

                    if metrics.get('success_perfect', False):
                        transmission_stats['success_perfect'] += 1
                    if metrics.get('success_good', False):
                        transmission_stats['success_good'] += 1
                    if metrics.get('success_usable', False):
                        transmission_stats['success_usable'] += 1
                    if metrics.get('success_degraded', False):
                        transmission_stats['success_degraded'] += 1

                    with resource_monitor.track_model_computation('yolo'):
                        detections = detector.detect(frame, img_path, original_size=original_size)

                    if success:
                        transmission_stats['successful'] += 1
                    else:
                        transmission_stats['failed'] += 1
                else:
                    with resource_monitor.track_model_computation('yolo'):
                        detections = detector.detect(frame, img_path)
            else:
                detections = detector.detect(frame, img_path)
        
        elif isinstance(detector, HDCWirelessDetector):
            is_yolo_frame = (detector.tracker.frame_count % detector.tracker.yolo_interval == 0)

            if is_yolo_frame:
                with resource_monitor.track_model_computation('yolo'):
                    detections, semantic_sim = detector.detect(frame, img_path)
            else:
                with resource_monitor.track_model_computation('hdc'):
                    detections, semantic_sim = detector.detect(frame, img_path)

            if semantic_sim is not None:
                transmission_stats['semantic_similarities'].append(semantic_sim)

            if transmitter is not None and detector.tracker.yolo_calls > 0:
                if detector.tracker.frame_count % detector.tracker.yolo_interval == 1:
                    tracker_state = detector.get_tracker_state()

                    with resource_monitor.track_transmission():
                        compressed_data, success, snr_db, latency_ms, received_state, metrics = transmitter.transmit(
                            tracker_state, current_distance, use_differential=True)

                    transmission_stats['total_transmissions'] += 1
                    transmission_stats['total_bytes'] += len(compressed_data)
                    transmission_stats['snr_db_samples'].append(snr_db)
                    transmission_stats['latency_ms_samples'].append(latency_ms)
                    transmission_stats['ber_samples'].append(metrics.get('ber', 0.0))
                    transmission_stats['bler_samples'].append(metrics.get('bler', 0.0))
                    transmission_stats['psnr_samples'].append(metrics.get('psnr', 0.0))
                    transmission_stats['ssim_samples'].append(metrics.get('ssim', 0.0))

                    if metrics.get('success_perfect', False):
                        transmission_stats['success_perfect'] += 1
                    if metrics.get('success_good', False):
                        transmission_stats['success_good'] += 1
                    if metrics.get('success_usable', False):
                        transmission_stats['success_usable'] += 1
                    if metrics.get('success_degraded', False):
                        transmission_stats['success_degraded'] += 1

                    if success:
                        transmission_stats['successful'] += 1
                    else:
                        transmission_stats['failed'] += 1
        
        if len(detections) == 0:
            frame_count += 1
            continue
        
        pred_boxes = np.array([d['bbox'] for d in detections])
        pred_cls = np.array([d['class'] for d in detections])
        confs = np.array([d['confidence'] for d in detections])
        
        correct = np.zeros(len(pred_boxes), dtype=bool)
        detected = np.zeros(len(gt_boxes), dtype=bool)
        target_cls = np.full(len(pred_boxes), -1, dtype=int)
        
        if len(gt_boxes) > 0:
            ious = box_iou(torch.tensor(pred_boxes).float(),
                          torch.tensor(gt_boxes).float()).numpy()
            
            for i in np.argsort(-confs):
                valid = np.where((gt_labels == pred_cls[i]) & (ious[i] > iou_thres) & (~detected))[0]
                
                if len(valid) > 0:
                    best_idx = valid[np.argmax(ious[i][valid])]
                    correct[i] = True
                    detected[best_idx] = True
                    target_cls[i] = gt_labels[best_idx]
        
        all_tp.append(correct)
        all_conf.append(confs)
        all_pred_cls.append(pred_cls)
        all_target_cls.append(target_cls)
        frame_count += 1
    
    elapsed_time = (time.time() - start_time) * 1000
    resource_stats = resource_monitor.stop()
    
    if len(all_tp) > 0:
        all_tp = np.concatenate(all_tp)
        all_conf = np.concatenate(all_conf)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_target_cls = np.concatenate(all_target_cls)
        
        metrics = compute_metrics(all_tp, all_conf, all_pred_cls, all_target_cls)
    else:
        metrics = {'map50': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    num_frames = len([f for f in os.listdir(sequence_path) if f.lower().endswith(('.jpg', '.png'))])
    fps = num_frames / (elapsed_time / 1000) if elapsed_time > 0 else 0
    
    return metrics, {
        'num_frames': num_frames, 'elapsed_time_ms': elapsed_time, 'fps': fps,
        'yolo_calls': detector.yolo_calls if hasattr(detector, 'yolo_calls') else detector.tracker.yolo_calls,
        'hdc_calls': detector.hdc_calls if hasattr(detector, 'hdc_calls') else detector.tracker.hdc_calls,
        'transmission_stats': transmission_stats, 'resource_stats': resource_stats
    }


def evaluate_system(model_path: str, data_path: str, experiment_name: str,
                   config: Dict, sequence_list: Optional[List[str]] = None,
                   baseline_stats: Optional[Dict] = None,
                   embedded_platform: str = 'nvidia_jetson_nano') -> ExperimentResult:
    sequences_path = os.path.join(data_path, 'sequences')
    annotations_path = os.path.join(data_path, 'annotations')

    yolo_model = YOLO(model_path)

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        yolo_model.to('mps')
        print("YOLO: MPS acceleration")
    elif torch.cuda.is_available():
        yolo_model.to('cuda')
        print("YOLO: CUDA acceleration")
    else:
        print("YOLO: CPU mode")
    
    system_type = config.get('system_type')
    yolo_interval = config.get('yolo_interval')
    
    channel_params = config.get('channel_params', {})
    carrier_frequency = channel_params.get('carrier_frequency', 2.4e9)
    drone_height_m = channel_params.get('drone_height_m', 50.0)
    drone_velocity_ms = channel_params.get('drone_velocity_ms', 10.0)
    tx_power_dbm = channel_params.get('tx_power_dbm', 20.0)
    scenario = channel_params.get('scenario', 'umi')
    enable_pathloss = channel_params.get('enable_pathloss', True)
    enable_shadow_fading = channel_params.get('enable_shadow_fading', True)
    
    # Create transmitter based on system type
    if system_type == 'baseline_wireless':
        detector = BaselineWirelessDetector(yolo_model)
        transmitter = FrameTransmitter3GPP(
            target_size=(256, 256),
            jpeg_quality=30,
            carrier_frequency=carrier_frequency,
            drone_height_m=drone_height_m,
            drone_velocity_ms=drone_velocity_ms,
            tx_power_dbm=tx_power_dbm,
            scenario=scenario,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading,
            num_bits_per_symbol=2
        )
    elif system_type == 'baseline_interval':
        detector = BaselineIntervalDetector(yolo_model, yolo_interval=yolo_interval)
        transmitter = FrameTransmitter3GPP(
            target_size=(256, 256),
            jpeg_quality=30,
            carrier_frequency=carrier_frequency,
            drone_height_m=drone_height_m,
            drone_velocity_ms=drone_velocity_ms,
            tx_power_dbm=tx_power_dbm,
            scenario=scenario,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading,
            num_bits_per_symbol=2
        )
    elif system_type == 'hdc_wireless':
        detector = HDCWirelessDetector(yolo_model, yolo_interval=yolo_interval)
        transmitter = HDCStateTransmitter3GPP(
            carrier_frequency=carrier_frequency,
            drone_height_m=drone_height_m,
            drone_velocity_ms=drone_velocity_ms,
            tx_power_dbm=tx_power_dbm,
            scenario=scenario,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading
        )
    else:
        raise ValueError(f"Unknown system type: {system_type}")

    if sequence_list is not None:
        all_sequences = sequence_list
    else:
        all_sequences = sorted([d for d in os.listdir(sequences_path)
                               if os.path.isdir(os.path.join(sequences_path, d)) and not d.startswith('.')])
    
    total_frames = 0
    total_yolo_calls = 0
    total_hdc_calls = 0
    
    map50_values, precision_values, recall_values, fps_values = [], [], [], []
    tx_success_rates, semantic_sims, bytes_transmitted = [], [], []
    snr_samples, latency_samples = [], []
    ber_samples, bler_samples, psnr_samples, ssim_samples = [], [], [], []
    total_success_perfect, total_success_good, total_success_usable, total_success_degraded = 0, 0, 0, 0
    all_energy_kwh, all_energy_wh, all_power_watts = [], [], []
    
    for seq in tqdm(all_sequences, desc=f"{experiment_name}"):
        seq_dir = os.path.join(sequences_path, seq)
        ann_file = os.path.join(annotations_path, seq + ".txt")
        
        if not os.path.exists(ann_file):
            continue
        
        if hasattr(detector, 'yolo_calls'):
            detector.yolo_calls = 0
            detector.hdc_calls = 0
        else:
            detector.tracker.yolo_calls = 0
            detector.tracker.hdc_calls = 0
        
        seq_monitor = ResourceMonitor(embedded_platform=embedded_platform)

        metrics, stats = evaluate_sequence(detector, seq_dir, ann_file, transmitter,
                                           channel_params, resource_monitor=seq_monitor,
                                           embedded_platform=embedded_platform)
        
        total_frames += stats['num_frames']
        total_yolo_calls += stats['yolo_calls']
        total_hdc_calls += stats['hdc_calls']
        
        map50_values.append(metrics['map50'])
        precision_values.append(metrics['precision'])
        recall_values.append(metrics['recall'])
        fps_values.append(stats['fps'])
        
        rs = stats['resource_stats']
        all_energy_kwh.append(rs['energy']['total_kwh'])
        all_energy_wh.append(rs['energy']['total_wh'])
        all_power_watts.append(rs['energy']['avg_power_watts'])
        
        if stats['transmission_stats']['total_transmissions'] > 0:
            tx_rate = stats['transmission_stats']['successful'] / stats['transmission_stats']['total_transmissions']
            tx_success_rates.append(tx_rate)

            avg_bytes = stats['transmission_stats']['total_bytes'] / stats['transmission_stats']['total_transmissions']
            bytes_transmitted.append(avg_bytes)

            snr_samples.extend(stats['transmission_stats']['snr_db_samples'])
            latency_samples.extend(stats['transmission_stats']['latency_ms_samples'])
            ber_samples.extend(stats['transmission_stats']['ber_samples'])
            bler_samples.extend(stats['transmission_stats']['bler_samples'])
            psnr_samples.extend(stats['transmission_stats']['psnr_samples'])
            ssim_samples.extend(stats['transmission_stats']['ssim_samples'])

            total_success_perfect += stats['transmission_stats']['success_perfect']
            total_success_good += stats['transmission_stats']['success_good']
            total_success_usable += stats['transmission_stats']['success_usable']
            total_success_degraded += stats['transmission_stats']['success_degraded']

        if stats['transmission_stats']['semantic_similarities']:
            semantic_sims.extend(stats['transmission_stats']['semantic_similarities'])

    transmission_dict = {}
    if tx_success_rates:
        transmission_dict['success_rate'] = float(np.mean(tx_success_rates))
        transmission_dict['semantic_similarity'] = float(np.mean(semantic_sims)) if semantic_sims else 0.0
        transmission_dict['avg_bytes'] = float(np.mean(bytes_transmitted)) if bytes_transmitted else 0.0
        transmission_dict['avg_snr_db'] = float(np.mean(snr_samples)) if snr_samples else 0.0
        transmission_dict['min_snr_db'] = float(np.min(snr_samples)) if snr_samples else 0.0
        transmission_dict['max_snr_db'] = float(np.max(snr_samples)) if snr_samples else 0.0
        transmission_dict['avg_latency_ms'] = float(np.mean(latency_samples)) if latency_samples else 0.0

        if transmission_dict['avg_latency_ms'] > 0 and transmission_dict['avg_bytes'] > 0:
            throughput_mbps = (transmission_dict['avg_bytes'] * 8) / (transmission_dict['avg_latency_ms'] / 1000) / 1e6
            transmission_dict['throughput_mbps'] = float(throughput_mbps)
        else:
            transmission_dict['throughput_mbps'] = 0.0

        transmission_dict['avg_ber'] = float(np.mean(ber_samples)) if ber_samples else 0.0
        transmission_dict['avg_bler'] = float(np.mean(bler_samples)) if bler_samples else 0.0
        transmission_dict['avg_psnr_db'] = float(np.mean(psnr_samples)) if psnr_samples else 0.0
        transmission_dict['avg_ssim'] = float(np.mean(ssim_samples)) if ssim_samples else 0.0

        total_tx = len(snr_samples)

        transmission_dict['tiered_success'] = {
            'perfect': {'count': int(total_success_perfect),
                       'percentage': float(100 * total_success_perfect / total_tx) if total_tx > 0 else 0.0},
            'good': {'count': int(total_success_good),
                    'percentage': float(100 * total_success_good / total_tx) if total_tx > 0 else 0.0},
            'usable': {'count': int(total_success_usable),
                      'percentage': float(100 * total_success_usable / total_tx) if total_tx > 0 else 0.0},
            'degraded': {'count': int(total_success_degraded),
                        'percentage': float(100 * total_success_degraded / total_tx) if total_tx > 0 else 0.0}
        }

        transmission_dict['packet_failure_rate'] = 1.0 - transmission_dict['success_rate']

        if system_type == 'hdc_wireless' and transmitter:
            transmission_dict['compression_ratio'] = float(transmitter.get_compression_ratio())

        transmission_dict['channel_stats'] = transmitter.get_channel_stats()
    else:
        transmission_dict = None
    
    accuracy_dict = {
        'map50': float(np.mean(map50_values)),
        'precision': float(np.mean(precision_values)),
        'recall': float(np.mean(recall_values))
    }
    
    total_energy_kwh = np.sum(all_energy_kwh)
    total_energy_wh = np.sum(all_energy_wh)
    avg_power_watts = np.mean(all_power_watts) if all_power_watts else 0

    efficiency_dict = {
        'energy': {
            'total_kwh': float(total_energy_kwh),
            'total_wh': float(total_energy_wh),
            'avg_power_watts': float(avg_power_watts)
        },
    
        'platform': embedded_platform,
        'fps': float(np.mean(fps_values)),
        'yolo_calls': int(total_yolo_calls),
        'hdc_calls': int(total_hdc_calls),
        'total_frames': int(total_frames)
    }
    
    if baseline_stats:
        baseline_energy = baseline_stats.get('energy', {}).get('total_kwh', 0)
        baseline_bytes = baseline_stats.get('avg_bytes', 0)

        if baseline_energy > 0:
            energy_savings = ((baseline_energy - total_energy_kwh) / baseline_energy) * 100
            efficiency_dict['energy_savings_pct'] = float(energy_savings)

        if system_type == 'hdc_wireless' and total_frames > 0:
            yolo_reduction = (1 - total_yolo_calls / total_frames) * 100
            efficiency_dict['yolo_reduction_pct'] = float(yolo_reduction)

        if transmission_dict and baseline_bytes > 0:
            bytes_savings = ((baseline_bytes - transmission_dict['avg_bytes']) / baseline_bytes) * 100
            transmission_dict['bandwidth_savings_pct'] = float(bytes_savings)
    
    return ExperimentResult(
        experiment_name=experiment_name,
        config=config,
        transmission=transmission_dict,
        accuracy=accuracy_dict,
        efficiency=efficiency_dict,
        total_frames=total_frames,
        timestamp=datetime.now().isoformat()
    )


def filter_config_constants(config: Dict) -> Dict:
    if 'channel_params' not in config:
        return config

    filtered_config = config.copy()
    channel_params = config['channel_params'].copy()

    constants_to_remove = ['carrier_frequency', 'tx_power_dbm', 'scenario',
                           'enable_pathloss', 'enable_shadow_fading']

    for const in constants_to_remove:
        channel_params.pop(const, None)

    filtered_config['channel_params'] = channel_params
    return filtered_config


def save_results(results: List[ExperimentResult], output_file: str,
                baseline_map50: Optional[float] = None) -> None:
    results_dict = []

    for r in results:
        r_dict = asdict(r)

        # Filter out constant config parameters
        r_dict['config'] = filter_config_constants(r_dict['config'])

        if baseline_map50 and r.config.get('system_type') != 'baseline_wireless':
            if baseline_map50 > 0:
                accuracy_loss = ((baseline_map50 - r.accuracy['map50']) / baseline_map50) * 100
                r_dict['accuracy']['accuracy_loss_pct'] = float(accuracy_loss)

        results_dict.append(r_dict)

    summary = {
        "simulation_type": "3gpp_link_level_drone_scenario",
        "channel_model": "Sionna TR 38.901 UMi (2.4GHz, 20dBm)",
        "experiments": results_dict
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results: {output_file}")




def print_comparison_table(results: List[ExperimentResult]) -> None:
    baseline = None
    for r in results:
        if r.config.get('system_type') == 'baseline_wireless':
            baseline = r
            break

    print("\n" + "="*220)
    print("3GPP DRONE CHANNEL: LINK-LEVEL SIMULATION")
    print("="*220)
    
    if baseline:
        platform = baseline.efficiency.get('platform', 'unknown')
        print(f"\nBASELINE WIRELESS (Large Packets - Full Frames) - Platform: {platform.upper().replace('_', ' ')}")
        print(f"   mAP50: {baseline.accuracy['map50']:.4f} | "
              f"Energy: {baseline.efficiency['energy']['total_wh']:.2f} Wh | "
              f"Power: {baseline.efficiency['energy']['avg_power_watts']:.2f} W | "
              f"FPS: {baseline.efficiency['fps']:.2f}")
        if baseline.transmission:
            print(f"   Transmission: Success={baseline.transmission['success_rate']*100:.1f}%, "
                  f"FAILED={baseline.transmission.get('packet_failure_rate', 0)*100:.1f}%, "
                  f"Avg Size={baseline.transmission['avg_bytes']/1000:.1f} KB")
            print(f"   Channel: SNR={baseline.transmission['avg_snr_db']:.1f} dB, "
                  f"Latency={baseline.transmission['avg_latency_ms']:.2f} ms")
    
    print(f"\n{'Experiment':<42} {'Energy (Wh)':<20} {'Power (W)':<20} {'Packet Size':<20} {'Accuracy':<15}")
    print("-"*220)

    # Get battery info if available
    platform = baseline.efficiency.get('platform', 'unknown') if baseline else 'unknown'
    battery_wh = 0
    base_flight_time = 0
    if platform in EMBEDDED_POWER_PROFILES:
        battery_wh = EMBEDDED_POWER_PROFILES[platform].get('battery_wh', 0)
        base_flight_time = EMBEDDED_POWER_PROFILES[platform].get('flight_time_min', 0)

    baseline_power = baseline.efficiency['energy']['avg_power_watts'] if baseline else 0

    for result in results:
        exp_name = result.experiment_name
        energy_wh = result.efficiency['energy']['total_wh']
        power_w = result.efficiency['energy']['avg_power_watts']

        if result.transmission:
            avg_bytes = result.transmission['avg_bytes']
            size_str = f"{avg_bytes/1000:.1f} KB"
        else:
            size_str = "N/A"

        accuracy_str = f"{result.accuracy['map50']:.4f}"

        energy_str = f"{energy_wh:.2f}"
        power_str = f"{power_w:.2f}"

        print(f"{exp_name:<42} {energy_str:<20} {power_str:<20} {size_str:<20} {accuracy_str:<15}")

    print("="*220)

    # Print energy savings analysis
    if baseline:
        print("\n" + "="*120)
        print("ENERGY SAVINGS ANALYSIS")
        print("="*120)

        for result in results:
            if result.config.get('system_type') == 'baseline_wireless':
                continue

            exp_name = result.experiment_name
            result_power = result.efficiency['energy']['avg_power_watts']

            if baseline_power > 0:
                power_savings_w = baseline_power - result_power
                power_savings_pct = (power_savings_w / baseline_power) * 100

                print(f"\n{exp_name}:")
                print(f"  Power Savings: {power_savings_w:.3f} W ({power_savings_pct:.1f}%)")

        print("="*120)

    print("="*220)


def run_interval_optimization(model_path: str, data_path: str,
                              embedded_platform: str = 'crazyflie_ai_deck',
                              n_sequences: int = 56) -> None:

    print(f"\n{'='*100}")
    print(f"INTERVAL OPTIMIZATION")
    print(f"{'='*100}\n")

    # Use moderate scenario for testing
    test_scenario = {
        'carrier_frequency': 2.4e9,
        'drone_height_m': 25.0,
        'drone_velocity_ms': 5.0,
        'tx_power_dbm': 20.0,
        'horizontal_dist_m': 125.0,
        'scenario': 'umi',
        'enable_pathloss': True,
        'enable_shadow_fading': True
    }

    # Test these intervals
    intervals = [1, 3, 5, 7, 10, 15, 20, 30]

    all_results = []

    for interval in intervals:
        print(f"\n{'#'*100}")
        print(f"# INTERVAL: {interval}")
        print(f"{'#'*100}\n")

        print(f"Baseline Interval {interval}...")
        baseline_interval_result = evaluate_system(
            model_path, data_path,
            experiment_name=f'Baseline_Interval_{interval}',
            config={'system_type': 'baseline_interval', 'yolo_interval': interval,
                   'channel_params': test_scenario},
            n_sequences=n_sequences,
            embedded_platform=embedded_platform
        )
        all_results.append(baseline_interval_result)

        print(f"HDC Wireless interval {interval}...")
        hdc_result = evaluate_system(
            model_path, data_path,
            experiment_name=f'HDC_Interval_{interval}',
            config={'system_type': 'hdc_wireless', 'yolo_interval': interval,
                   'channel_params': test_scenario},
            n_sequences=n_sequences,
            embedded_platform=embedded_platform
        )
        all_results.append(hdc_result)

    output_file = "results/interval_optimization_results.json"
    save_results(all_results, output_file)

    print(f"\n{'='*120}")
    print(f"RESULTS")
    print(f"{'='*120}")
    print(f"\n{'Interval':<10} {'System':<20} {'mAP@50':<12} {'Energy (Wh)':<15} "
          f"{'Throughput':<15} {'Bandwidth':<15} {'YOLO Calls':<12}")
    print("-"*120)

    for result in all_results:
        interval_str = result.experiment_name.split('_')[-1]
        system = 'Baseline' if 'Baseline' in result.experiment_name else 'HDC'

        map50 = result.accuracy['map50']
        energy_wh = result.efficiency['energy']['total_wh']
        throughput = result.transmission.get('throughput_mbps', 0) if result.transmission else 0
        bandwidth = result.transmission.get('avg_bytes', 0) if result.transmission else 0
        yolo_calls = result.efficiency['yolo_calls']

        print(f"{interval_str:<10} {system:<20} {map50:<12.4f} {energy_wh:<15.2f} "
              f"{throughput:<15.2f} {bandwidth/1000:<15.1f} {yolo_calls:<12}")

    print("="*120)

    print(f"\n{'='*120}")
    print("HDC vs Baseline Interval Analysis")
    print(f"{'='*120}")
    print("BaselineIntervalDetector: YOLO every N frames, static detections between")
    print("HDCWirelessDetector: YOLO every N frames, HDC tracking updates between")
    print("="*120)

    print(f"\nResults: {output_file}\n")


def main():
    quick_test = False

    model_path = "yolo_nano.pt"
    data_path = "data/VisDrone2019-VID"
    embedded_platform = 'crazyflie_ai_deck'
    yolo_interval = 10

    print(f"\n{'='*100}")
    print(f"CRAZYFLIE AI-DECK EVALUATION")
    if quick_test:
        print(f"QUICK TEST MODE")
    print(f"{'='*100}")
    print(f"Platform: {embedded_platform.upper().replace('_', ' ')}")
    if embedded_platform in EMBEDDED_POWER_PROFILES:
        profile = EMBEDDED_POWER_PROFILES[embedded_platform]
        print(f"TDP: {profile['typical_tdp']:.2f}W | Battery: {profile['battery_wh']:.3f}Wh | Flight: {profile.get('flight_time_min', 'N/A')}min")
    print(f"YOLO Interval: {yolo_interval}")
    print(f"{'='*100}\n")

    scenarios = [
        {
            'name': 'scenario_A_sparse_highway',
            'description': 'Sparse Highway - Good Channel',
            'use_case': 'Minimal traffic, excellent conditions - both systems should perform similarly',
            'sequences': [
                'uav0000218_00001_v',  # 6.1 avg objects - very sparse
            ],
            'params': {
                'carrier_frequency': 2.4e9,
                'drone_height_m': 20.0,  # Low altitude for good LOS
                'drone_velocity_ms': 2.0,  # Slow movement
                'tx_power_dbm': 20.0,
                'horizontal_dist_m': 50.0,  # Close range = excellent channel
                'scenario': 'umi',
                'enable_pathloss': True,
                'enable_shadow_fading': True
            }
        },
        {
            'name': 'scenario_B_ultra_dense',
            'description': 'Ultra-Dense Intersection - Overwhelming Complexity',
            'use_case': 'Extremely crowded scene - both systems should struggle equally',
            'sequences': [
                'uav0000138_00000_v',  # 151 avg objects - extremely dense
            ],
            'params': {
                'carrier_frequency': 2.4e9,
                'drone_height_m': 40.0,  # High altitude
                'drone_velocity_ms': 8.0,  # Fast movement
                'tx_power_dbm': 20.0,
                'horizontal_dist_m': 200.0,  # Far range = poor channel
                'scenario': 'umi',
                'enable_pathloss': True,
                'enable_shadow_fading': True
            }
        },
        {
            'name': 'scenario_C_low_traffic',
            'description': 'Low-Traffic Road - Simple Scene',
            'use_case': 'Minimal dynamics, good channel - interval should be sufficient',
            'sequences': [
                'uav0000145_00000_v',  # 11.2 avg objects - low traffic
            ],
            'params': {
                'carrier_frequency': 2.4e9,
                'drone_height_m': 18.0,  # Low-medium altitude
                'drone_velocity_ms': 3.0,  # Moderate movement
                'tx_power_dbm': 20.0,
                'horizontal_dist_m': 60.0,  # Close-medium range
                'scenario': 'umi',
                'enable_pathloss': True,
                'enable_shadow_fading': True
            }
        }
    ]

    # ===========================================================================
    # RUN ALL SCENARIOS
    # ===========================================================================
    for scenario_idx, scenario_config in enumerate(scenarios, 1):
        scenario_name = scenario_config['name']
        scenario_desc = scenario_config['description']
        use_case = scenario_config['use_case']
        channel_params = scenario_config['params']
        sequence_list = scenario_config.get('sequences', None)

        if quick_test and sequence_list:
            sequence_list = [sequence_list[0]]
            print(f"\nQuick test: First sequence from {scenario_name}")

        print(f"\n{'#'*100}")
        print(f"# SCENARIO {scenario_idx}/{len(scenarios)}: {scenario_desc}")
        print(f"# {use_case}")
        print(f"# Sequences: {len(sequence_list)}" if sequence_list else "# Sequences: All")
        print(f"# H={channel_params['drone_height_m']}m, "
              f"D={channel_params['horizontal_dist_m']}m, "
              f"V={channel_params['drone_velocity_ms']}m/s, "
              f"TX={channel_params['tx_power_dbm']}dBm")
        print(f"{'#'*100}\n")

        scenario_output_dir = f"results/"
        os.makedirs(scenario_output_dir, exist_ok=True)

        all_results = []

        print("\n" + "="*80)
        print("Baseline Wireless")
        print("="*80)
        baseline_result = evaluate_system(
            model_path, data_path,
            experiment_name=f"Baseline_Every_Frame_{scenario_name}",
            config={'system_type': 'baseline_wireless', 'channel_params': channel_params},
            sequence_list=sequence_list,
            embedded_platform=embedded_platform
        )
        all_results.append(baseline_result)

        baseline_stats = {
            'energy': baseline_result.efficiency['energy'],
            'avg_bytes': baseline_result.transmission['avg_bytes'] if baseline_result.transmission else 0
        }
        baseline_map50 = baseline_result.accuracy['map50']

        print("\n" + "="*80)
        print(f"Baseline Interval ({yolo_interval} frames)")
        print("="*80)
        baseline_interval_result = evaluate_system(
            model_path, data_path,
            experiment_name=f'Baseline_Interval_{yolo_interval}_{scenario_name}',
            config={'system_type': 'baseline_interval', 'yolo_interval': yolo_interval, 'channel_params': channel_params},
            sequence_list=sequence_list,
            baseline_stats=baseline_stats,
            embedded_platform=embedded_platform
        )
        all_results.append(baseline_interval_result)

        print("\n" + "="*80)
        print(f"HDC Wireless (interval: {yolo_interval})")
        print("="*80)
        hdc_result = evaluate_system(
            model_path, data_path,
            experiment_name=f'HDC_State_TX_Int{yolo_interval}_{scenario_name}',
            config={'system_type': 'hdc_wireless', 'yolo_interval': yolo_interval, 'channel_params': channel_params},
            sequence_list=sequence_list,
            baseline_stats=baseline_stats,
            embedded_platform=embedded_platform
        )
        all_results.append(hdc_result)

        print_comparison_table(all_results)

        output_file = f"{scenario_output_dir}/{scenario_name}_results.json"
        save_results(all_results, output_file, baseline_map50)

        print(f"\n{'#'*100}")
        print(f"# SCENARIO {scenario_idx}/{len(scenarios)} COMPLETE")
        print(f"# Results: {scenario_output_dir}/")
        print(f"{'#'*100}\n")

        if quick_test and (scenario_idx == 1):
            print(f"\nQUICK TEST: Stopping after scenario 1\n")
            break


    print(f"\n{'='*100}")
    if quick_test:
        print(f"QUICK TEST COMPLETE")
    else:
        print(f"ALL {len(scenarios)} SCENARIOS COMPLETE")
    print(f"Results: results/")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()