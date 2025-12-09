"""
Drone detection and tracking module.
"""

import os
import warnings
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import torch
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from sionna.channel.tr38901 import UMi, UMa, AntennaArray
from sionna.channel import GenerateOFDMChannel, OFDMChannel
from sionna.ofdm import ResourceGrid
from sionna.mimo import StreamManagement


class Sionna3GPPDroneChannel:

    def __init__(self,
                 carrier_frequency: float = 2.4e9,
                 subcarrier_spacing: float = 30e3,
                 num_subcarriers: int = 64,
                 num_ofdm_symbols: int = 14,
                 drone_height_m: float = 50.0,
                 drone_velocity_ms: float = 10.0,
                 ground_station_height_m: float = 10.0,
                 scenario: str = 'umi',
                 batch_size: int = 1,
                 enable_pathloss: bool = True,
                 enable_shadow_fading: bool = True):

        self.carrier_frequency = carrier_frequency
        self.subcarrier_spacing = subcarrier_spacing
        self.num_subcarriers = num_subcarriers
        self.num_ofdm_symbols = num_ofdm_symbols
        self.drone_height_m = drone_height_m
        self.drone_velocity_ms = drone_velocity_ms
        self.ground_station_height_m = ground_station_height_m
        self.scenario = scenario.lower()
        self.batch_size = batch_size
        self.enable_pathloss = enable_pathloss
        self.enable_shadow_fading = enable_shadow_fading

        self.c = 3e8
        self.wavelength = self.c / carrier_frequency
        self.max_doppler_hz = drone_velocity_ms / self.wavelength
        self.bandwidth = num_subcarriers * subcarrier_spacing

        self.ut_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=carrier_frequency
        )

        self.bs_array = AntennaArray(
            num_rows=1, num_cols=1,
            polarization='single', polarization_type='V',
            antenna_pattern='omni', carrier_frequency=carrier_frequency
        )

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=num_subcarriers,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=1, num_streams_per_tx=1,
            cyclic_prefix_length=6,
            pilot_pattern='kronecker',
            pilot_ofdm_symbol_indices=[2, 11]
        )

        self.rx_tx_association = np.array([[1]])
        self.stream_management = StreamManagement(self.rx_tx_association, 1)

        self._create_channel_model()

        self.snr_history = []
        self.doppler_history = []
        self.los_history = []

    def _create_channel_model(self):
        common_params = {
            'carrier_frequency': self.carrier_frequency,
            'ut_array': self.ut_array,
            'bs_array': self.bs_array,
            'direction': 'uplink',
            'enable_pathloss': self.enable_pathloss,
            'enable_shadow_fading': self.enable_shadow_fading,
        }

        if self.scenario == 'umi':
            self.channel_model = UMi(o2i_model='low', **common_params)
        elif self.scenario == 'uma':
            self.channel_model = UMa(o2i_model='low', **common_params)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")

        self.ofdm_channel = OFDMChannel(
            self.channel_model, self.resource_grid,
            add_awgn=True, normalize_channel=True, return_channel=True
        )

    def set_topology(self, horizontal_dist_m: float):
        ut_loc = tf.constant(
            [[[horizontal_dist_m, 0.0, self.drone_height_m]]],
            dtype=tf.float32
        )
        ut_loc = tf.tile(ut_loc, [self.batch_size, 1, 1])

        bs_loc = tf.constant(
            [[[0.0, 0.0, self.ground_station_height_m]]],
            dtype=tf.float32
        )
        bs_loc = tf.tile(bs_loc, [self.batch_size, 1, 1])

        ut_orientations = tf.zeros([self.batch_size, 1, 3], dtype=tf.float32)
        bs_orientations = tf.zeros([self.batch_size, 1, 3], dtype=tf.float32)

        velocity_angle = np.random.uniform(0, 2 * np.pi)
        vx = self.drone_velocity_ms * np.cos(velocity_angle)
        vy = self.drone_velocity_ms * np.sin(velocity_angle)

        ut_velocities = tf.constant([[[vx, vy, 0.0]]], dtype=tf.float32)
        ut_velocities = tf.tile(ut_velocities, [self.batch_size, 1, 1])

        in_state = tf.zeros([self.batch_size, 1], dtype=tf.bool)

        self.channel_model.set_topology(
            ut_loc=ut_loc, bs_loc=bs_loc,
            ut_orientations=ut_orientations, bs_orientations=bs_orientations,
            ut_velocities=ut_velocities, in_state=in_state, los=None
        )

        doppler = self.drone_velocity_ms / self.wavelength
        self.doppler_history.append(doppler)

        d_3d = np.sqrt(horizontal_dist_m**2 +
                      (self.drone_height_m - self.ground_station_height_m)**2)

        return d_3d, doppler

    def generate_channel(self):
        channel_gen = GenerateOFDMChannel(self.channel_model, self.resource_grid)
        h_freq = channel_gen(self.batch_size)
        return h_freq

    def get_channel_stats(self) -> Dict:
        """Return only variable channel statistics useful for visualization."""
        return {
            'avg_snr_db': float(np.mean(self.snr_history)) if self.snr_history else 0.0,
            'min_snr_db': float(np.min(self.snr_history)) if self.snr_history else 0.0,
            'max_snr_db': float(np.max(self.snr_history)) if self.snr_history else 0.0,
            'std_snr_db': float(np.std(self.snr_history)) if self.snr_history else 0.0,
            'num_samples': len(self.snr_history)
        }


class HDCTracker:
    """Hyperdimensional Computing based object tracker."""

    def __init__(self, dim: int = 10000, yolo_interval: int = 10):
        self.dim = dim
        self.yolo_interval = yolo_interval

        np.random.seed(42)
        self.appearance_basis = self._random_hv()
        self.color_basis = self._random_hv()
        self.location_basis = self._random_hv()

        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0
        self.last_yolo_detections = []

        self.yolo_calls = 0
        self.hdc_calls = 0

    def _random_hv(self) -> np.ndarray:
        return np.random.choice([-1, 1], size=self.dim)

    def _bipolarize(self, vec: np.ndarray) -> np.ndarray:
        return np.where(vec >= 0, 1, -1)

    def encode_appearance(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return self._random_hv()

        patch = cv2.resize(patch, (32, 32))
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)

        encoded = np.zeros(self.dim)
        for i, weight in enumerate(hist):
            shift = int((i / 16) * self.dim)
            encoded += weight * np.roll(self.appearance_basis, shift)

        return self._bipolarize(encoded)

    def encode_color(self, patch: np.ndarray) -> np.ndarray:
        if patch.size == 0:
            return self._random_hv()

        patch = cv2.resize(patch, (32, 32))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        hist_h = hist_h / (hist_h.sum() + 1e-8)

        encoded = np.zeros(self.dim)
        for i, weight in enumerate(hist_h):
            shift = int((i / 12) * self.dim)
            encoded += weight * np.roll(self.color_basis, shift)

        return self._bipolarize(encoded)

    def encode_location(self, cx: float, cy: float, img_w: int, img_h: int) -> np.ndarray:
        cx_norm = cx / img_w
        cy_norm = cy / img_h

        shift_x = int(cx_norm * self.dim * 0.5)
        shift_y = int(cy_norm * self.dim * 0.5)

        loc_hv = np.roll(self.location_basis, shift_x)
        loc_hv = np.roll(loc_hv, shift_y)

        return loc_hv

    def encode_object(self, frame: np.ndarray, bbox: List[float], cls: int = 0) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return self._random_hv()

        patch = frame[y1:y2, x1:x2]

        appearance_hv = self.encode_appearance(patch)
        color_hv = self.encode_color(patch)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        location_hv = self.encode_location(cx, cy, w, h)

        object_hv = appearance_hv * color_hv * location_hv

        return object_hv

    def compute_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        return np.dot(hv1, hv2) / self.dim

    def predict_bbox(self, bbox: List[float], velocity: List[float]) -> List[float]:
        x1, y1, x2, y2 = bbox
        vx, vy = velocity
        return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]

    def initialize_tracks(self, frame: np.ndarray, detections: List[Dict]) -> None:
        self.tracks = {}
        self.next_track_id = 0
        self.last_yolo_detections = detections.copy()

        for det in detections:
            object_hv = self.encode_object(frame, det['bbox'], det['class'])

            self.tracks[self.next_track_id] = {
                'hv': object_hv,
                'bbox': det['bbox'],
                'class': det['class'],
                'confidence': det['confidence'],
                'velocity': [0.0, 0.0],
                'age': 0
            }
            self.next_track_id += 1

    def hdc_track(self, frame: np.ndarray) -> List[Dict]:
        tracked_detections = []
        tracks_to_remove = []
        h, w = frame.shape[:2]

        for track_id, track in self.tracks.items():
            pred_bbox = self.predict_bbox(track['bbox'], track['velocity'])

            pred_bbox[0] = max(0, min(w - 10, pred_bbox[0]))
            pred_bbox[1] = max(0, min(h - 10, pred_bbox[1]))
            pred_bbox[2] = max(10, min(w, pred_bbox[2]))
            pred_bbox[3] = max(10, min(h, pred_bbox[3]))

            curr_hv = self.encode_object(frame, pred_bbox, track['class'])
            similarity = self.compute_similarity(curr_hv, track['hv'])

            if similarity > 0.65:
                old_cx = (track['bbox'][0] + track['bbox'][2]) / 2
                old_cy = (track['bbox'][1] + track['bbox'][3]) / 2
                new_cx = (pred_bbox[0] + pred_bbox[2]) / 2
                new_cy = (pred_bbox[1] + pred_bbox[3]) / 2

                new_vx = new_cx - old_cx
                new_vy = new_cy - old_cy
                track['velocity'][0] = 0.8 * track['velocity'][0] + 0.2 * new_vx
                track['velocity'][1] = 0.8 * track['velocity'][1] + 0.2 * new_vy

                tracked_detections.append({
                    'bbox': pred_bbox,
                    'class': track['class'],
                    'confidence': track['confidence'] * similarity
                })

                track['bbox'] = pred_bbox
                track['age'] += 1
                track['hv'] = self._bipolarize(0.9 * track['hv'] + 0.1 * curr_hv)
            else:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return tracked_detections

    def should_run_yolo(self) -> bool:
        run_yolo = self.frame_count % self.yolo_interval == 0
        self.frame_count += 1
        return run_yolo

    def compute_semantic_similarity(self, current_detections: List[Dict]) -> float:
        if not self.last_yolo_detections or not current_detections:
            return 0.0

        yolo_boxes = np.array([d['bbox'] for d in self.last_yolo_detections])
        curr_boxes = np.array([d['bbox'] for d in current_detections])

        if len(yolo_boxes) == 0 or len(curr_boxes) == 0:
            return 0.0

        try:
            ious = box_iou(torch.tensor(curr_boxes).float(),
                          torch.tensor(yolo_boxes).float()).numpy()

            if ious.size > 0:
                semantic_sim = ious.max(axis=1).mean()
            else:
                semantic_sim = 0.0
        except Exception:
            semantic_sim = 0.0

        return float(semantic_sim)

    def get_state_for_transmission(self) -> Dict:
        state_data = []

        for track_id, track in self.tracks.items():
            track_data = {
                'id': track_id,
                'position': ((track['bbox'][0] + track['bbox'][2]) / 2,
                            (track['bbox'][1] + track['bbox'][3]) / 2),
                'bbox_size': (track['bbox'][2] - track['bbox'][0],
                            track['bbox'][3] - track['bbox'][1]),
                'velocity': track['velocity'],
                'class': track['class'],
                'confidence': track['confidence'],
                'age': track['age']
            }
            state_data.append(track_data)

        return {
            'frame_count': self.frame_count,
            'next_track_id': self.next_track_id,
            'tracks': state_data
        }

    def restore_from_transmission(self, state_data: Dict) -> None:
        self.frame_count = state_data['frame_count']
        self.next_track_id = state_data['next_track_id']

        self.tracks = {}
        for track_data in state_data['tracks']:
            cx, cy = track_data['position']
            w, h = track_data['bbox_size']

            bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

            velocity = track_data['velocity']
            if isinstance(velocity, np.ndarray):
                velocity = velocity.tolist()
            elif not isinstance(velocity, list):
                velocity = list(velocity)

            self.tracks[track_data['id']] = {
                'hv': self._random_hv(),
                'bbox': bbox,
                'class': track_data['class'],
                'confidence': track_data['confidence'],
                'velocity': velocity,
                'age': track_data['age']
            }


class BaselineWirelessDetector:
    """Baseline: Transmit full frames and run YOLO at receiver every frame."""

    def __init__(self, yolo_model: YOLO, imgsz: int = 640, conf_threshold: float = 0.25):
        self.yolo = yolo_model
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.yolo_calls = 0
        self.hdc_calls = 0

    def detect(self, frame: np.ndarray, img_path: str,
               original_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
        h, w = frame.shape[:2]

        results = self.yolo(source=frame, imgsz=max(h, w), verbose=False,
                           stream=False, augment=False, conf=self.conf_threshold)[0]

        self.yolo_calls += 1

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            if original_size is not None:
                orig_h, orig_w = original_size
                scale_x = orig_w / w
                scale_y = orig_h / h

                for bbox, cls, conf in zip(boxes, classes, confs):
                    scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y,
                                  bbox[2] * scale_x, bbox[3] * scale_y]
                    detections.append({'bbox': scaled_bbox, 'class': cls, 'confidence': float(conf)})
            else:
                for bbox, cls, conf in zip(boxes, classes, confs):
                    detections.append({'bbox': bbox.tolist(), 'class': cls, 'confidence': float(conf)})

        return detections


class BaselineIntervalDetector:
    """Baseline: Transmit frames every N frames, run YOLO, repeat last detection."""

    def __init__(self, yolo_model: YOLO, yolo_interval: int = 10,
                 imgsz: int = 640, conf_threshold: float = 0.25):
        self.yolo = yolo_model
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.yolo_interval = yolo_interval
        self.frame_count = 0
        self.yolo_calls = 0
        self.hdc_calls = 0
        self.last_detections = []

    def should_run_yolo(self) -> bool:
        run_yolo = self.frame_count % self.yolo_interval == 0
        self.frame_count += 1
        return run_yolo

    def detect(self, frame: np.ndarray, img_path: str,
               original_size: Optional[Tuple[int, int]] = None) -> List[Dict]:
        if self.should_run_yolo():
            h, w = frame.shape[:2]

            results = self.yolo(source=frame, imgsz=max(h, w), verbose=False,
                               stream=False, augment=False, conf=self.conf_threshold)[0]

            self.yolo_calls += 1

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                if original_size is not None:
                    orig_h, orig_w = original_size
                    scale_x = orig_w / w
                    scale_y = orig_h / h

                    for bbox, cls, conf in zip(boxes, classes, confs):
                        scaled_bbox = [bbox[0] * scale_x, bbox[1] * scale_y,
                                      bbox[2] * scale_x, bbox[3] * scale_y]
                        detections.append({'bbox': scaled_bbox, 'class': cls, 'confidence': float(conf)})
                else:
                    for bbox, cls, conf in zip(boxes, classes, confs):
                        detections.append({'bbox': bbox.tolist(), 'class': cls, 'confidence': float(conf)})

            self.last_detections = detections
            return detections
        else:
            return self.last_detections.copy()


class HDCWirelessDetector:
    """YOLO detector with HDC temporal tracking."""

    def __init__(self, yolo_model: YOLO, yolo_interval: int = 10,
                 imgsz: int = 640, conf_threshold: float = 0.25):
        self.yolo = yolo_model
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.tracker = HDCTracker(yolo_interval=yolo_interval)

    def detect(self, frame: np.ndarray, img_path: str) -> Tuple[List[Dict], Optional[float]]:
        semantic_sim = None

        if self.tracker.should_run_yolo():
            results = self.yolo(source=frame, imgsz=self.imgsz, verbose=False,
                               stream=False, augment=False, conf=self.conf_threshold)[0]

            self.tracker.yolo_calls += 1

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for bbox, cls, conf in zip(boxes, classes, confs):
                    detections.append({'bbox': bbox.tolist(), 'class': cls, 'confidence': float(conf)})

            self.tracker.initialize_tracks(frame, detections)
        else:
            detections = self.tracker.hdc_track(frame)
            self.tracker.hdc_calls += 1
            semantic_sim = self.tracker.compute_semantic_similarity(detections)

        return detections, semantic_sim

    def get_tracker_state(self) -> Dict:
        return self.tracker.get_state_for_transmission()

    def restore_tracker_state(self, state_data: Dict) -> None:
        self.tracker.restore_from_transmission(state_data)
