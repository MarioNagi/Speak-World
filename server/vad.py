# server/vad.py - Enhanced version with better speech detection
import collections
import webrtcvad
import logging
from typing import Optional

log = logging.getLogger("v2v.vad")

class VADChunker:
    """Enhanced VAD-based chunker with better speech detection and noise handling.

    - Feed 20ms frames (bytes) via add_frame().
    - When voiced segment ends, returns bytes() of the full segment.
    - Improved with pre-speech detection and better noise handling.
    """
    
    def __init__(self, sample_rate=16000, frame_ms=20, aggressiveness=2, 
                 end_silence_ms=500, pre_speech_frames=3, min_speech_frames=10):
        """
        Initialize VAD chunker.
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_ms: Frame duration in milliseconds (10, 20, or 30)
            aggressiveness: VAD aggressiveness level (0-3, higher = more aggressive)
            end_silence_ms: Silence duration to end a segment
            pre_speech_frames: Number of frames to include before speech starts
            min_speech_frames: Minimum frames required for a valid speech segment
        """
        assert frame_ms in (10, 20, 30), f"frame_ms must be 10, 20, or 30, got {frame_ms}"
        assert sample_rate in (8000, 16000, 32000, 48000), f"Unsupported sample rate: {sample_rate}"
        assert 0 <= aggressiveness <= 3, f"aggressiveness must be 0-3, got {aggressiveness}"
        
        self.sr = sample_rate
        self.bytes_per_sample = 2  # PCM16
        self.frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * self.bytes_per_sample
        self.frame_ms = frame_ms
        
        try:
            self.vad = webrtcvad.Vad(aggressiveness)
        except Exception as e:
            log.error(f"Failed to initialize WebRTC VAD: {e}")
            raise
        
        # State tracking
        self.in_segment = False
        self.current = bytearray()
        
        # Timing parameters
        self.silence_needed = max(1, end_silence_ms // frame_ms)
        self.silence_count = 0
        self.pre_speech_frames = pre_speech_frames
        self.min_speech_frames = min_speech_frames
        self.speech_frame_count = 0
        
        # Ring buffer for pre-speech frames
        self.pre_speech_buffer = collections.deque(maxlen=pre_speech_frames)
        
        # Statistics
        self.total_frames = 0
        self.speech_frames = 0
        self.segments_produced = 0
        
        log.info(f"VAD initialized: sr={sample_rate}, frame_ms={frame_ms}, "
                f"aggressiveness={aggressiveness}, end_silence_ms={end_silence_ms}")
    
    def add_frame(self, frame_bytes: bytes) -> Optional[bytes]:
        """
        Add a frame of audio data.
        
        Args:
            frame_bytes: PCM16 audio data for one frame
            
        Returns:
            Complete speech segment bytes if a segment just ended, None otherwise
        """
        if len(frame_bytes) != self.frame_bytes:
            log.debug(f"Frame size mismatch: expected {self.frame_bytes}, got {len(frame_bytes)}")
            return None
        
        self.total_frames += 1
        
        try:
            is_speech = self.vad.is_speech(frame_bytes, self.sr)
        except Exception as e:
            log.warning(f"VAD analysis failed for frame: {e}")
            return None
        
        if is_speech:
            self.speech_frames += 1
        
        # State machine for segment detection
        if not self.in_segment:
            # Not in segment - waiting for speech
            if is_speech:
                log.debug("Speech detected - starting new segment")
                self.in_segment = True
                self.silence_count = 0
                self.speech_frame_count = 1
                
                # Add buffered pre-speech frames
                for buffered_frame in self.pre_speech_buffer:
                    self.current.extend(buffered_frame)
                
                # Add current speech frame
                self.current.extend(frame_bytes)
                
                # Clear the buffer since we used it
                self.pre_speech_buffer.clear()
            else:
                # Still no speech - add to ring buffer for potential future use
                self.pre_speech_buffer.append(frame_bytes)
        else:
            # In segment - collecting audio
            self.current.extend(frame_bytes)
            
            if is_speech:
                self.silence_count = 0
                self.speech_frame_count += 1
            else:
                self.silence_count += 1
                
                # Check if we've had enough silence to end the segment
                if self.silence_count >= self.silence_needed:
                    # Only produce segment if we had enough speech
                    if self.speech_frame_count >= self.min_speech_frames:
                        segment = bytes(self.current)
                        duration_ms = len(self.current) / self.bytes_per_sample / self.sr * 1000
                        
                        log.debug(f"Segment complete: {duration_ms:.0f}ms, "
                                f"{self.speech_frame_count} speech frames")
                        
                        self._reset_segment_state()
                        self.segments_produced += 1
                        return segment
                    else:
                        log.debug(f"Segment too short: only {self.speech_frame_count} speech frames "
                                f"(min: {self.min_speech_frames})")
                        self._reset_segment_state()
        
        return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush any remaining audio in the current segment.
        
        Returns:
            Remaining segment bytes if any, None otherwise
        """
        if self.current and self.speech_frame_count >= self.min_speech_frames:
            segment = bytes(self.current)
            duration_ms = len(segment) / self.bytes_per_sample / self.sr * 1000
            
            log.debug(f"Flushed segment: {duration_ms:.0f}ms, {self.speech_frame_count} speech frames")
            
            self._reset_segment_state()
            self.segments_produced += 1
            return segment
        
        if self.current:
            log.debug(f"Discarded short segment on flush: {self.speech_frame_count} speech frames")
        
        self._reset_segment_state()
        return None
    
    def _reset_segment_state(self):
        """Reset internal state for new segment detection."""
        self.in_segment = False
        self.current = bytearray()
        self.silence_count = 0
        self.speech_frame_count = 0
        self.pre_speech_buffer.clear()
    
    def get_stats(self) -> dict:
        """Get VAD processing statistics."""
        if self.total_frames == 0:
            return {"status": "no_frames_processed"}
        
        speech_ratio = self.speech_frames / self.total_frames * 100
        processing_time_s = self.total_frames * self.frame_ms / 1000
        
        return {
            "total_frames": self.total_frames,
            "speech_frames": self.speech_frames,
            "speech_ratio_percent": round(speech_ratio, 1),
            "segments_produced": self.segments_produced,
            "processing_time_seconds": round(processing_time_s, 1),
            "current_segment_active": self.in_segment,
            "current_segment_frames": len(self.current) // self.frame_bytes if self.current else 0
        }
    
    def reset(self):
        """Reset all internal state and statistics."""
        self._reset_segment_state()
        self.total_frames = 0
        self.speech_frames = 0
        self.segments_produced = 0
        log.debug("VAD state reset")