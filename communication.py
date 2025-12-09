import pickle
import zlib
from typing import Dict, Tuple, List, Optional

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import cv2
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.utils import ebnodb2no
from drone import Sionna3GPPDroneChannel


def calculate_psnr(original: np.ndarray, received: np.ndarray, max_pixel: float = 255.0) -> float:
    if original is None or received is None:
        return 0.0

    if original.shape != received.shape:
        received = cv2.resize(received, (original.shape[1], original.shape[0]))

    mse = np.mean((original.astype(float) - received.astype(float)) ** 2)
    if mse == 0:
        return 100.0

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(original: np.ndarray, received: np.ndarray) -> float:
    if original is None or received is None:
        return 0.0

    if original.shape != received.shape:
        received = cv2.resize(received, (original.shape[1], original.shape[0]))

    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        received_gray = cv2.cvtColor(received, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        received_gray = received

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(original_gray.astype(float), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(received_gray.astype(float), (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(original_gray.astype(float) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(received_gray.astype(float) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original_gray.astype(float) * received_gray.astype(float), (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


class Drone3GPPChannel:

    def __init__(self,
                 channel: Sionna3GPPDroneChannel,
                 tx_power_dbm: float = 20.0,
                 noise_figure_db: float = 6.0,
                 num_bits_per_symbol: int = 4,
                 k: int = 512,
                 n: int = 1024):
        self.channel = channel
        self.tx_power_dbm = tx_power_dbm
        self.noise_figure_db = noise_figure_db
        self.num_bits_per_symbol = num_bits_per_symbol
        self.k = k
        self.n = n
        self.coderate = k / n

        self.encoder = LDPC5GEncoder(k, n)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

        self.constellation = Constellation("qam", num_bits_per_symbol, trainable=False)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper("app", constellation=self.constellation)

        self.successful_transmissions = 0
        self.total_transmissions = 0
        self.bler_history = []

        self.channel_cache = {}
        self.cache_size = 50

    @tf.function(reduce_retracing=True)
    def _encode_modulate(self, b):
        c = self.encoder(b)
        x = self.mapper(c)
        return x

    @tf.function(reduce_retracing=True)
    def _demodulate_decode(self, y, no):
        llr = self.demapper([y, no])
        b_hat = self.decoder(llr)
        return b_hat

    def transmit_receive(self, data_bytes: bytes,
                        horizontal_dist_m: float) -> Tuple[Optional[bytes], bool, Dict]:

        self.total_transmissions += 1

        try:
            dist_key = round(horizontal_dist_m / 10) * 10

            if dist_key in self.channel_cache:
                h_freq, d_3d, doppler, channel_power_gain = self.channel_cache[dist_key]
            else:
                d_3d, doppler = self.channel.set_topology(horizontal_dist_m)
                h_freq = self.channel.generate_channel()

                h_effective = h_freq[0, 0, 0, 0, 0, :, :]

                channel_power_gain = tf.reduce_mean(tf.abs(h_effective)**2).numpy()

                if len(self.channel_cache) >= self.cache_size:
                    self.channel_cache.pop(next(iter(self.channel_cache)))
                self.channel_cache[dist_key] = (h_freq, d_3d, doppler, channel_power_gain)

            tx_power_linear = 10**(self.tx_power_dbm / 10) / 1000

            k_boltzmann = 1.38e-23
            T = 290
            noise_power_linear = k_boltzmann * T * self.channel.bandwidth
            noise_figure_linear = 10**(self.noise_figure_db / 10)
            total_noise_power = noise_power_linear * noise_figure_linear

            if not self.channel.enable_pathloss:
                # Free space path loss formula
                path_loss_db = 20 * np.log10(d_3d) + 20 * np.log10(self.channel.carrier_frequency) - 147.55
                path_loss_linear = 10**(-path_loss_db / 10)
                rx_signal_power = tx_power_linear * channel_power_gain * path_loss_linear
            else:
                # Path loss already included in channel
                rx_signal_power = tx_power_linear * channel_power_gain

            snr_linear = rx_signal_power / total_noise_power
            snr_db = 10 * np.log10(snr_linear + 1e-12)

            self.channel.snr_history.append(snr_db)

            spectral_efficiency = self.coderate * self.num_bits_per_symbol
            ebno_db = snr_db - 10 * np.log10(spectral_efficiency)
            no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)

            data_bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
            num_bits = len(data_bits)
            num_codewords = int(np.ceil(num_bits / self.k))
            padding_bits = num_codewords * self.k - num_bits

            if padding_bits > 0:
                data_bits = np.concatenate([data_bits, np.zeros(padding_bits, dtype=np.uint8)])

            info_bits = data_bits.reshape(1, num_codewords, self.k)
            b = tf.constant(info_bits, dtype=tf.float32)

            x = self._encode_modulate(b)

            num_symbols_per_cw = self.n // self.num_bits_per_symbol
            total_symbols = num_codewords * num_symbols_per_cw
            x_flat = tf.reshape(x, [1, total_symbols])

            h_effective = h_freq[0, 0, 0, 0, 0, :, :]
            h_flat = tf.reshape(h_effective, [-1])
            num_available_re = tf.shape(h_flat)[0]

            if total_symbols <= num_available_re:
                h_used = h_flat[:total_symbols]
            else:
                num_reps = (total_symbols + num_available_re - 1) // num_available_re
                h_used = tf.tile(h_flat, [num_reps])[:total_symbols]

            h_used = tf.cast(h_used, dtype=x_flat.dtype)

            avg_channel_power = tf.reduce_mean(tf.abs(h_used)**2)
            normalization_factor = tf.cast(tf.sqrt(avg_channel_power + 1e-10), dtype=h_used.dtype)
            h_normalized = h_used / normalization_factor

            y_before_noise = x_flat * h_normalized[tf.newaxis, :]

            noise_std = tf.sqrt(no / 2.0)
            noise = tf.complex(
                tf.random.normal(tf.shape(y_before_noise), mean=0.0, stddev=noise_std, dtype=tf.float32),
                tf.random.normal(tf.shape(y_before_noise), mean=0.0, stddev=noise_std, dtype=tf.float32)
            )
            y_with_noise = y_before_noise + noise

            h_expanded = h_normalized[tf.newaxis, :]
            y_equalized = y_with_noise / (h_expanded + tf.cast(1e-10, dtype=h_expanded.dtype))

            h_norm_mag_sq = tf.abs(h_normalized)**2
            h_norm_mag_sq_clipped = tf.maximum(h_norm_mag_sq, 0.01)
            avg_noise_enhancement = tf.reduce_mean(1.0 / h_norm_mag_sq_clipped)
            no_eff = no * avg_noise_enhancement

            y_reshaped = tf.reshape(y_equalized, [1, num_codewords, num_symbols_per_cw])
            b_hat = self._demodulate_decode(y_reshaped, no_eff)

            b_hat_np = b_hat.numpy()
            b_np = b.numpy()

            codeword_errors = ~np.all(b_hat_np == b_np, axis=2)[0]
            num_failed_codewords = int(np.sum(codeword_errors))
            actual_bler = num_failed_codewords / num_codewords

            b_hat_flat = b_hat_np.flatten()
            if padding_bits > 0:
                b_hat_flat = b_hat_flat[:-padding_bits]

            decoded_bytes = np.packbits(b_hat_flat.astype(np.uint8)).tobytes()[:len(data_bytes)]

            original_bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
            decoded_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))[:len(original_bits)]
            bit_errors = np.sum(original_bits != decoded_bits)
            ber = bit_errors / len(original_bits) if len(original_bits) > 0 else 0

            self.bler_history.append(actual_bler)

            success = (actual_bler < 0.2) and (ber < 0.1)

            if success:
                self.successful_transmissions += 1

            metrics = {
                'snr_db': float(snr_db),
                'ber': float(ber),
                'bler': float(actual_bler),
                'ebno_db': float(ebno_db),
                'bit_errors': int(bit_errors),
                'doppler_hz': float(doppler),
                'distance_3d_m': float(d_3d),
                'num_codewords': int(num_codewords),
                'num_failed_codewords': int(num_failed_codewords),
                'actual_bler': float(actual_bler),
                'channel_power_gain_db': float(10 * np.log10(channel_power_gain + 1e-12))
            }

            return decoded_bytes, success, metrics

        except Exception as e:
            self.bler_history.append(1)
            failed_snr = float(np.mean(self.channel.snr_history[-10:])) if len(self.channel.snr_history) > 0 else 0.0
            return None, False, {'error': str(e), 'snr_db': failed_snr, 'ber': 1.0, 'bler': 1.0}

    def get_stats(self) -> Dict:
        success_rate = self.successful_transmissions / max(self.total_transmissions, 1)
        avg_bler = float(np.mean(self.bler_history)) if self.bler_history else 0.0

        return {
            'success_rate': success_rate,
            'avg_bler': avg_bler,
            'total_transmissions': self.total_transmissions
        }


class FrameTransmitter3GPP:

    def __init__(self,
                 target_size: Tuple[int, int] = (320, 320),
                 jpeg_quality: int = 30,
                 carrier_frequency: float = 2.4e9,
                 drone_height_m: float = 50.0,
                 drone_velocity_ms: float = 10.0,
                 tx_power_dbm: float = 20.0,
                 scenario: str = 'umi',
                 enable_pathloss: bool = True,
                 enable_shadow_fading: bool = True,
                 num_bits_per_symbol: int = 4):
        self.target_size = target_size
        self.jpeg_quality = jpeg_quality

        self.channel = Sionna3GPPDroneChannel(
            carrier_frequency=carrier_frequency,
            drone_height_m=drone_height_m,
            drone_velocity_ms=drone_velocity_ms,
            scenario=scenario,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading
        )

        self.link_sim = Drone3GPPChannel(
            channel=self.channel,
            tx_power_dbm=tx_power_dbm,
            num_bits_per_symbol=num_bits_per_symbol,
            k=512, n=1024
        )

        self.transmission_sizes = []
        self.latencies = []

    def compress_frame(self, frame: np.ndarray) -> bytes:
        resized = cv2.resize(frame, self.target_size)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, encoded_img = cv2.imencode('.jpg', resized, encode_param)
        compressed_bytes = encoded_img.tobytes()
        self.transmission_sizes.append(len(compressed_bytes))
        return compressed_bytes

    def estimate_transmission_time_ms(self, num_bytes: int) -> float:
        bits_per_ofdm_frame = (self.channel.num_subcarriers *
                               self.channel.num_ofdm_symbols *
                               self.link_sim.num_bits_per_symbol *
                               self.link_sim.coderate)

        symbol_duration_s = 1.0 / self.channel.subcarrier_spacing
        cp_duration_s = symbol_duration_s * (6.0 / self.channel.num_subcarriers)
        total_symbol_duration_s = symbol_duration_s + cp_duration_s

        num_bits = num_bytes * 8
        num_ofdm_frames_needed = np.ceil(num_bits / bits_per_ofdm_frame)

        tx_time_ms = (num_ofdm_frames_needed * self.channel.num_ofdm_symbols *
                     total_symbol_duration_s * 1000)

        overhead_ms = 1.0

        return tx_time_ms + overhead_ms

    def transmit(self, frame: np.ndarray,
                horizontal_dist_m: float = 100.0) -> Tuple[bytes, bool, float, float, Optional[np.ndarray], Dict]:
        compressed_data = self.compress_frame(frame)

        decoded_bytes, success, metrics = self.link_sim.transmit_receive(
            compressed_data, horizontal_dist_m
        )

        latency_ms = self.estimate_transmission_time_ms(len(compressed_data))
        self.latencies.append(latency_ms)

        snr_db = metrics.get('snr_db', 0.0)
        ber = metrics.get('ber', 1.0)

        received_frame = None
        psnr = 0.0
        ssim = 0.0

        if decoded_bytes is not None:
            try:
                nparr = np.frombuffer(decoded_bytes, np.uint8)
                received_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if received_frame is not None:
                    received_frame = cv2.resize(received_frame, self.target_size)
                    original_resized = cv2.resize(frame, self.target_size)
                    psnr = calculate_psnr(original_resized, received_frame)
                    ssim = calculate_ssim(original_resized, received_frame)
            except Exception:
                received_frame = None
                success = False

        success_perfect = (ber < 0.01)
        success_good = (ber >= 0.01 and ber < 0.05)
        success_usable = (ber >= 0.05 and ber < 0.10)
        success_degraded = (ber >= 0.10 and ber < 0.20)

        metrics['psnr'] = float(psnr)
        metrics['ssim'] = float(ssim)
        metrics['success_perfect'] = success_perfect
        metrics['success_good'] = success_good
        metrics['success_usable'] = success_usable
        metrics['success_degraded'] = success_degraded

        return compressed_data, success, snr_db, latency_ms, received_frame, metrics

    def get_avg_transmission_size(self) -> float:
        return np.mean(self.transmission_sizes) if self.transmission_sizes else 0.0

    def get_channel_stats(self) -> Dict:
        return self.channel.get_channel_stats()

    def get_stats(self) -> Dict:
        return self.link_sim.get_stats()


class HDCStateTransmitter3GPP:

    def __init__(self,
                 carrier_frequency: float = 2.4e9,
                 drone_height_m: float = 50.0,
                 drone_velocity_ms: float = 10.0,
                 tx_power_dbm: float = 20.0,
                 scenario: str = 'umi',
                 enable_pathloss: bool = True,
                 enable_shadow_fading: bool = True):
        self.channel = Sionna3GPPDroneChannel(
            carrier_frequency=carrier_frequency,
            drone_height_m=drone_height_m,
            drone_velocity_ms=drone_velocity_ms,
            scenario=scenario,
            enable_pathloss=enable_pathloss,
            enable_shadow_fading=enable_shadow_fading
        )

        self.link_sim = Drone3GPPChannel(
            channel=self.channel,
            tx_power_dbm=tx_power_dbm,
            k=512, n=1024
        )

        self.previous_state = None
        self.uncompressed_sizes = []
        self.compressed_sizes = []
        self.latencies = []

    def compress_state(self, tracker_state: Dict, use_differential: bool = True) -> bytes:
        if use_differential and self.previous_state is not None:
            transmission_data = self._compute_differential(tracker_state)
        else:
            transmission_data = self._prepare_full_state(tracker_state)

        self.previous_state = tracker_state

        serialized = pickle.dumps(transmission_data)
        compressed = zlib.compress(serialized, level=9)

        self.uncompressed_sizes.append(len(serialized))
        self.compressed_sizes.append(len(compressed))

        return compressed

    def _prepare_full_state(self, tracker_state: Dict) -> Dict:
        transmission_data = {
            'type': 'full',
            'frame_count': tracker_state['frame_count'],
            'next_track_id': tracker_state['next_track_id'],
            'tracks': []
        }

        for track in tracker_state['tracks']:
            if 'position' not in track or 'bbox_size' not in track or 'velocity' not in track:
                continue

            velocity = np.array(track['velocity'], dtype=np.float32)

            track_compressed = {
                'id': track['id'],
                'position': (int(np.round(track['position'][0] * 2)),
                           int(np.round(track['position'][1] * 2))),
                'bbox_size': (int(np.round(track['bbox_size'][0] * 2)),
                            int(np.round(track['bbox_size'][1] * 2))),
                'velocity': tuple(int(np.round(v * 20)) for v in velocity),
                'class': track['class'],
                'confidence': int(track['confidence'] * 100),
                'age': track['age']
            }
            transmission_data['tracks'].append(track_compressed)

        return transmission_data

    def _compute_differential(self, current_state: Dict) -> Dict:
        prev_tracks = {t['id']: t for t in self.previous_state['tracks']}
        curr_tracks = {t['id']: t for t in current_state['tracks']}

        new_ids = set(curr_tracks.keys()) - set(prev_tracks.keys())
        removed_ids = set(prev_tracks.keys()) - set(curr_tracks.keys())
        common_ids = set(curr_tracks.keys()) & set(prev_tracks.keys())

        diff_data = {
            'type': 'differential',
            'frame_count': current_state['frame_count'],
            'new_tracks': [],
            'removed_ids': list(removed_ids),
            'updated_tracks': []
        }

        for track_id in new_ids:
            track = curr_tracks[track_id]
            if 'position' not in track or 'bbox_size' not in track or 'velocity' not in track:
                continue

            velocity = np.array(track['velocity'], dtype=np.float32)
            diff_data['new_tracks'].append({
                'id': track['id'],
                'position': (int(np.round(track['position'][0] * 2)),
                           int(np.round(track['position'][1] * 2))),
                'bbox_size': (int(np.round(track['bbox_size'][0] * 2)),
                            int(np.round(track['bbox_size'][1] * 2))),
                'velocity': tuple(int(np.round(v * 20)) for v in velocity),
                'class': track['class'],
                'confidence': int(track['confidence'] * 100),
                'age': track['age']
            })

        for track_id in common_ids:
            prev_track = prev_tracks[track_id]
            curr_track = curr_tracks[track_id]

            pos_changed = not np.allclose(prev_track['position'], curr_track['position'], atol=0.5)
            size_changed = not np.allclose(prev_track['bbox_size'], curr_track['bbox_size'], atol=0.5)

            if pos_changed or size_changed:
                curr_vel = np.array(curr_track['velocity'], dtype=np.float32)
                update = {'id': track_id}

                if pos_changed:
                    update['position'] = (int(np.round(curr_track['position'][0] * 2)),
                                        int(np.round(curr_track['position'][1] * 2)))
                if size_changed:
                    update['bbox_size'] = (int(np.round(curr_track['bbox_size'][0] * 2)),
                                          int(np.round(curr_track['bbox_size'][1] * 2)))

                update['velocity'] = tuple(int(np.round(v * 20)) for v in curr_vel)
                update['confidence'] = int(curr_track['confidence'] * 100)
                update['age'] = curr_track['age']

                diff_data['updated_tracks'].append(update)

        return diff_data

    def estimate_transmission_time_ms(self, num_bytes: int) -> float:
        bits_per_ofdm_frame = (self.channel.num_subcarriers *
                               self.channel.num_ofdm_symbols *
                               self.link_sim.num_bits_per_symbol *
                               self.link_sim.coderate)

        symbol_duration_s = 1.0 / self.channel.subcarrier_spacing
        cp_duration_s = symbol_duration_s * (6.0 / self.channel.num_subcarriers)
        total_symbol_duration_s = symbol_duration_s + cp_duration_s

        num_bits = num_bytes * 8
        num_ofdm_frames_needed = np.ceil(num_bits / bits_per_ofdm_frame)

        tx_time_ms = (num_ofdm_frames_needed * self.channel.num_ofdm_symbols *
                     total_symbol_duration_s * 1000)

        overhead_ms = 0.5

        return tx_time_ms + overhead_ms

    def get_compression_ratio(self) -> float:
        if not self.compressed_sizes:
            return 1.0
        return np.mean(self.uncompressed_sizes) / np.mean(self.compressed_sizes)

    def transmit(self, tracker_state: Dict, horizontal_dist_m: float = 100.0,
                use_differential: bool = True) -> Tuple[bytes, bool, float, float, Optional[Dict], Dict]:
        compressed_data = self.compress_state(tracker_state, use_differential)

        decoded_bytes, success, metrics = self.link_sim.transmit_receive(
            compressed_data, horizontal_dist_m
        )

        latency_ms = self.estimate_transmission_time_ms(len(compressed_data))
        self.latencies.append(latency_ms)

        snr_db = metrics.get('snr_db', 0.0)
        ber = metrics.get('ber', 1.0)

        received_state = None
        if success and decoded_bytes is not None:
            try:
                decompressed = zlib.decompress(decoded_bytes)
                transmission_data = pickle.loads(decompressed)

                if transmission_data['type'] == 'differential':
                    received_state = self._apply_differential(transmission_data)
                else:
                    received_state = transmission_data
            except Exception:
                received_state = None
                success = False

        success_perfect = (ber < 0.01)
        success_good = (ber >= 0.01 and ber < 0.05)
        success_usable = (ber >= 0.05 and ber < 0.10)
        success_degraded = (ber >= 0.10 and ber < 0.20)

        metrics['psnr'] = 0.0
        metrics['ssim'] = 0.0
        metrics['success_perfect'] = success_perfect
        metrics['success_good'] = success_good
        metrics['success_usable'] = success_usable
        metrics['success_degraded'] = success_degraded

        return compressed_data, success, snr_db, latency_ms, received_state, metrics

    def _apply_differential(self, diff_data: Dict) -> Dict:
        if self.previous_state is None:
            raise ValueError("Cannot apply differential without previous state")

        prev_tracks = {t['id']: t for t in self.previous_state['tracks']}

        for track_id in diff_data['removed_ids']:
            if track_id in prev_tracks:
                del prev_tracks[track_id]

        for new_track in diff_data['new_tracks']:
            prev_tracks[new_track['id']] = {
                'id': new_track['id'],
                'position': (new_track['position'][0] / 2.0, new_track['position'][1] / 2.0),
                'bbox_size': (new_track['bbox_size'][0] / 2.0, new_track['bbox_size'][1] / 2.0),
                'velocity': [v / 20.0 for v in new_track['velocity']],
                'class': new_track['class'],
                'confidence': new_track['confidence'] / 100.0,
                'age': new_track['age']
            }

        for update in diff_data['updated_tracks']:
            track_id = update['id']
            if track_id in prev_tracks:
                track = prev_tracks[track_id]
                if 'position' in update:
                    track['position'] = (update['position'][0] / 2.0, update['position'][1] / 2.0)
                if 'bbox_size' in update:
                    track['bbox_size'] = (update['bbox_size'][0] / 2.0, update['bbox_size'][1] / 2.0)
                if 'velocity' in update:
                    track['velocity'] = [v / 20.0 for v in update['velocity']]
                if 'confidence' in update:
                    track['confidence'] = update['confidence'] / 100.0
                if 'age' in update:
                    track['age'] = update['age']

        return {
            'type': 'full',
            'frame_count': diff_data['frame_count'],
            'next_track_id': self.previous_state['next_track_id'],
            'tracks': list(prev_tracks.values())
        }

    def get_channel_stats(self) -> Dict:
        return self.channel.get_channel_stats()

    def get_stats(self) -> Dict:
        return self.link_sim.get_stats()
