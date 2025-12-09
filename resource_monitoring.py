
import time
from typing import Dict, Optional
from contextlib import contextmanager


# Embedded system power profiles (Watts)
EMBEDDED_POWER_PROFILES = {
    'crazyflie_ai_deck': {
        # Crazyflie 2.1+ with AI-deck (GAP8 + ESP32)
        # References:
        # - STM32F405: ~150mW active (Cortex-M4 @ 168MHz)
        # - nRF51822: ~20mW active + ~80mW TX @ 20dBm
        # - GAP8: ~50mW idle, ~200-400mW during inference
        # - ESP32: ~160mW active
        # - Camera: ~50mW
        'idle_watts': 0.25,           # STM32 + nRF51 + GAP8 idle + ESP32 sleep
        'cpu_active_watts': 0.4,      # STM32 + nRF51 + ESP32 + Camera active
        'yolo_inference_watts': 1.2,  # GAP8 @ full load for YOLO Nano inference
        'hdc_encoding_watts': 0.15,   # Lightweight HDC encoding on STM32
        'transmission_watts': 0.1,    # nRF radio TX @ 20dBm (2.4GHz)
        'typical_tdp': 1.8,           # Total system power (excluding motors)
        'battery_wh': 0.925,          # 250mAh @ 3.7V
        'flight_time_min': 7,
    }
}


class ResourceMonitor:

    def __init__(self, embedded_platform: str = 'crazyfly', use_codecarbon: bool = True):
        self.start_time = None
        self.embedded_platform = embedded_platform
        self.use_codecarbon = use_codecarbon
        self.codecarbon_tracker = None
        self.total_energy_kwh = 0.0

        if self.use_codecarbon:
            try:
                from codecarbon import OfflineEmissionsTracker
                self.codecarbon_tracker = OfflineEmissionsTracker(
                    save_to_file=False,
                    save_to_api=False,
                    log_level='error',
                    tracking_mode='process',
                    country_iso_code='USA'
                )
            except Exception:
                self.use_codecarbon = False

        self.operation_counts = {
            'yolo_inferences': 0,
            'hdc_encodings': 0,
            'transmissions': 0
        }

        self.operation_durations = {
            'yolo_total_seconds': 0.0,
            'hdc_total_seconds': 0.0,
            'transmission_total_seconds': 0.0
        }

    def start(self):
        self.start_time = time.time()
        if self.use_codecarbon and self.codecarbon_tracker is not None:
            try:
                self.codecarbon_tracker.start()
            except Exception:
                self.use_codecarbon = False

    def log_operation(self, operation_type: str, count: int = 1):
        if operation_type in self.operation_counts:
            self.operation_counts[operation_type] += count

    @contextmanager
    def track_model_computation(self, operation_type: str):
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            if operation_type == 'yolo':
                self.operation_counts['yolo_inferences'] += 1
                self.operation_durations['yolo_total_seconds'] += duration
            elif operation_type == 'hdc':
                self.operation_counts['hdc_encodings'] += 1
                self.operation_durations['hdc_total_seconds'] += duration

    @contextmanager
    def track_transmission(self):
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.operation_counts['transmissions'] += 1
            self.operation_durations['transmission_total_seconds'] += duration

    def estimate_embedded_power(self, elapsed_time: float) -> Dict:
        if self.embedded_platform not in EMBEDDED_POWER_PROFILES:
            return {'total_wh': 0, 'avg_power_watts': 0, 'breakdown': {}, 'method': 'none'}

        profile = EMBEDDED_POWER_PROFILES[self.embedded_platform]

        tracked_time = (
            self.operation_durations['yolo_total_seconds'] +
            self.operation_durations['hdc_total_seconds'] +
            self.operation_durations['transmission_total_seconds']
        )

        if self.use_codecarbon and self.total_energy_kwh > 0:
            total_measured_wh = self.total_energy_kwh * 1000

            yolo_weight = self.operation_durations['yolo_total_seconds'] * profile['yolo_inference_watts']
            hdc_weight = self.operation_durations['hdc_total_seconds'] * profile['hdc_encoding_watts']
            trans_weight = self.operation_durations['transmission_total_seconds'] * profile['transmission_watts']
            base_time = max(0, elapsed_time - tracked_time)
            base_weight = base_time * profile['cpu_active_watts']

            total_weight = yolo_weight + hdc_weight + trans_weight + base_weight

            if total_weight > 0:
                yolo_energy_wh = total_measured_wh * (yolo_weight / total_weight)
                hdc_energy_wh = total_measured_wh * (hdc_weight / total_weight)
                transmission_energy_wh = total_measured_wh * (trans_weight / total_weight)
                active_energy_wh = total_measured_wh * (base_weight / total_weight)
            else:
                yolo_energy_wh = hdc_energy_wh = transmission_energy_wh = active_energy_wh = 0

            method = 'codecarbon'
        else:
            yolo_energy_wh = (profile['yolo_inference_watts'] *
                             self.operation_durations['yolo_total_seconds'] / 3600)

            hdc_energy_wh = (profile['hdc_encoding_watts'] *
                            self.operation_durations['hdc_total_seconds'] / 3600)

            transmission_energy_wh = (profile['transmission_watts'] *
                                     self.operation_durations['transmission_total_seconds'] / 3600)

            base_time = max(0, elapsed_time - tracked_time)
            active_energy_wh = profile['cpu_active_watts'] * (base_time / 3600)

            method = 'real_durations'

        total_energy_wh = yolo_energy_wh + hdc_energy_wh + transmission_energy_wh + active_energy_wh
        avg_power_watts = total_energy_wh / (elapsed_time / 3600) if elapsed_time > 0 else 0

        return {
            'total_wh': total_energy_wh,
            'avg_power_watts': avg_power_watts,
            'breakdown': {
                'yolo_wh': yolo_energy_wh,
                'hdc_wh': hdc_energy_wh,
                'transmission_wh': transmission_energy_wh,
                'base_active_wh': active_energy_wh
            },
            'method': method,
            'durations': {
                'yolo_avg_seconds': (self.operation_durations['yolo_total_seconds'] /
                                     self.operation_counts['yolo_inferences']
                                     if self.operation_counts['yolo_inferences'] > 0 else 0),
                'hdc_avg_seconds': (self.operation_durations['hdc_total_seconds'] /
                                    self.operation_counts['hdc_encodings']
                                    if self.operation_counts['hdc_encodings'] > 0 else 0),
                'transmission_avg_seconds': (self.operation_durations['transmission_total_seconds'] /
                                             self.operation_counts['transmissions']
                                             if self.operation_counts['transmissions'] > 0 else 0)
            }
        }

    def stop(self) -> Dict:
        elapsed_time = time.time() - self.start_time

        if self.use_codecarbon and self.codecarbon_tracker is not None:
            try:
                self.total_energy_kwh = self.codecarbon_tracker.stop()
                if self.total_energy_kwh is None or self.total_energy_kwh <= 0:
                    self.total_energy_kwh = 0.0
                    self.use_codecarbon = False
            except Exception:
                self.total_energy_kwh = 0.0
                self.use_codecarbon = False

        embedded_estimate = self.estimate_embedded_power(elapsed_time)

        return {
            'elapsed_time_seconds': elapsed_time,
            'energy': {
                'total_kwh': embedded_estimate['total_wh'] / 1000,
                'total_wh': embedded_estimate['total_wh'],
                'avg_power_watts': embedded_estimate['avg_power_watts'],
                'breakdown': embedded_estimate['breakdown'],
                'measurement_method': embedded_estimate['method']
            },
            'platform': self.embedded_platform,
            'operations': self.operation_counts.copy(),
            'durations': embedded_estimate['durations']
        }

    def get_energy_comparison(self, energy_stats: Dict) -> Dict:
        breakdown = energy_stats['breakdown']

        yolo_wh = breakdown['yolo_wh']
        hdc_wh = breakdown['hdc_wh']
        transmission_wh = breakdown['transmission_wh']

        total_model_energy = yolo_wh + hdc_wh
        total_energy = total_model_energy + transmission_wh

        if total_energy == 0:
            return {
                'model_computation_percentage': 0,
                'transmission_percentage': 0,
                'method': 'no_data'
            }

        return {
            'model_computation_wh': total_model_energy,
            'transmission_wh': transmission_wh,
            'total_wh': total_energy,
            'model_computation_percentage': (total_model_energy / total_energy) * 100,
            'transmission_percentage': (transmission_wh / total_energy) * 100,
            'yolo_wh': yolo_wh,
            'hdc_wh': hdc_wh,
            'method': 'real_durations'
        }
