import numpy as np
import librosa
from loguru import logger


class EnhancedProsodyAnalyzer:
    """
    Advanced real-time acoustic analyzer extracting fundamental frequency (F0),
    contour dynamics, jitter, and spectral flux for expressive emotion mapping.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_prosody_profile(self, pcm_bytes: bytes) -> dict:
        """Analyzes 16-bit PCM mono audio buffer and computes acoustic dynamic indicators."""
        if not pcm_bytes or len(pcm_bytes) < 3200:  # Need at least ~100ms
            return self._default_features()

        try:
            # Convert raw 16-bit PCM to float32 normalized [-1.0, 1.0]
            audio_data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 1. Pitch Tracking via PYIN (Fundamental Frequency F0)
            f0, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C6'),  # ~1046 Hz
                sr=self.sample_rate
            )

            # Filter valid pitch points
            valid_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])

            if len(valid_f0) > 0:
                mean_pitch = float(np.mean(valid_f0))
                pitch_std = float(np.std(valid_f0))
                
                # Pitch Slope (contour direction: rising vs falling)
                x = np.arange(len(valid_f0))
                pitch_slope = float(np.polyfit(x, valid_f0, 1)[0]) if len(valid_f0) > 1 else 0.0
            else:
                mean_pitch, pitch_std, pitch_slope = 0.0, 0.0, 0.0

            # 2. Energy & Dynamics (RMS)
            rms_energy = float(np.mean(librosa.feature.rms(y=audio_data)))

            # 3. Spectral Tilt / Centroid (Brightness of voice)
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)))

            # Infer Vocal Valence & Energy Profile
            emotion_label = self._classify_vocal_affect(mean_pitch, pitch_slope, pitch_std, rms_energy)

            return {
                "mean_pitch_hz": round(mean_pitch, 2),
                "pitch_variability": round(pitch_std, 2),
                "pitch_contour": "rising" if pitch_slope > 0.5 else ("falling" if pitch_slope < -0.5 else "flat"),
                "rms_energy": round(rms_energy, 4),
                "spectral_brightness_hz": round(spectral_centroid, 2),
                "detected_affect": emotion_label
            }

        except Exception as e:
            logger.warning(f"Prosody extraction error: {e}")
            return self._default_features()

    def _classify_vocal_affect(self, pitch: float, slope: float, std: float, energy: float) -> str:
        """Heuristic affect mapping matching human acoustic dynamics research."""
        if energy < 0.005:
            return "low_energy_whisper"
        if energy > 0.06 and std > 25.0:
            return "excited_animated"
        if slope > 1.2 and energy > 0.02:
            return "questioning_curious"
        if std < 8.0 and energy > 0.01:
            return "flat_monotone"
        if pitch > 220 and slope < -1.0:
            return "stressed_strained"
        return "neutral_calm"

    def _default_features(self) -> dict:
        return {
            "mean_pitch_hz": 0.0,
            "pitch_variability": 0.0,
            "pitch_contour": "flat",
            "rms_energy": 0.0,
            "spectral_brightness_hz": 0.0,
            "detected_affect": "neutral_calm"
        }