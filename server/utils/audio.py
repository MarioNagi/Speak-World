# utils/audio.py - Enhanced version
import numpy as np
import librosa
import logging

log = logging.getLogger("v2v.audio")

def pcm16_bytes_to_float32_mono(buf):
    """Convert PCM16 bytes to float32 mono audio with validation."""
    if not buf or len(buf) == 0:
        return np.array([], dtype=np.float32)
    
    if len(buf) % 2 != 0:
        log.warning(f"PCM16 buffer length {len(buf)} is not even, truncating")
        buf = buf[:-1]
    
    try:
        a = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        return a
    except Exception as e:
        log.error(f"Failed to convert PCM16 buffer: {e}")
        return np.array([], dtype=np.float32)

def float32_to_pcm16_bytes(x):
    """Convert float32 audio to PCM16 bytes with validation."""
    if x is None or len(x) == 0:
        return b''
    
    try:
        # Ensure we have a numpy array
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        
        # Clip to valid range and convert
        x_clipped = np.clip(x, -1.0, 1.0)
        pcm16 = (x_clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()
    except Exception as e:
        log.error(f"Failed to convert to PCM16: {e}")
        return b''

def resample(x, orig_sr, target_sr):
    """Resample audio with validation and error handling."""
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)
    
    if orig_sr == target_sr:
        return x.astype(np.float32)
    
    try:
        resampled = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)
        return resampled.astype(np.float32)
    except Exception as e:
        log.error(f"Resampling failed from {orig_sr} to {target_sr}: {e}")
        return x.astype(np.float32)  # Return original as fallback

def validate_audio_segment(audio_data, min_duration_ms=100, sample_rate=16000):
    """
    Validate if audio segment is worth processing.
    
    Args:
        audio_data: numpy array of audio samples
        min_duration_ms: minimum duration in milliseconds
        sample_rate: sample rate of audio
    
    Returns:
        bool: True if segment is valid for processing
    """
    if audio_data is None or len(audio_data) == 0:
        return False
    
    duration_ms = (len(audio_data) / sample_rate) * 1000
    if duration_ms < min_duration_ms:
        log.debug(f"Audio segment too short: {duration_ms:.1f}ms < {min_duration_ms}ms")
        return False
    
    # Check if audio has sufficient energy (not just silence)
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms < 0.001:  # Very quiet threshold
        log.debug(f"Audio segment too quiet: RMS={rms:.6f}")
        return False
    
    return True

def normalize_audio(audio_data, target_lufs=-23.0):
    """
    Normalize audio to target loudness (basic version).
    For production, consider using pyloudnorm library.
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    
    # Simple RMS-based normalization
    rms = np.sqrt(np.mean(audio_data ** 2))
    if rms > 0:
        # Convert target LUFS to linear scale (approximation)
        target_rms = 10 ** (target_lufs / 20)
        normalized = audio_data * (target_rms / rms)
        return np.clip(normalized, -1.0, 1.0)
    
    return audio_data

def apply_noise_gate(audio_data, threshold_db=-40.0):
    """Apply simple noise gate to reduce background noise."""
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    
    # Convert threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Apply gate
    mask = np.abs(audio_data) > threshold_linear
    gated_audio = audio_data * mask.astype(np.float32)
    
    return gated_audio